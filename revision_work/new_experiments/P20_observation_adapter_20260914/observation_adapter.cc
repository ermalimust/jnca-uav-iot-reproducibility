// P20 independent observation adapter. SPDX-License-Identifier: GPL-2.0-only
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/neighbor-cache-helper.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <array>
#include <cmath>
#include <algorithm>
using namespace ns3;

static double beginS=6,endS=18;
static uint32_t mainNode=1,monitorNode=2;
static Mac48Address apAddress,mainAddress;
static std::ofstream states,frames,apps;
static std::vector<Ptr<Socket>> senders,sinks;
static uint32_t payload=1472;
static double offeredMbps=200;
static std::vector<uint64_t> sends,socketFails,receives,receivedBytes;

class PilotLoss : public PropagationLossModel {
 public:
  static TypeId GetTypeId(){static TypeId id=TypeId("ns3::P20PilotLoss").SetParent<PropagationLossModel>().AddConstructor<PilotLoss>();return id;}
  double reference=40,exponent=3;
  std::vector<double> shadow;
  void Configure(double ref,double eta,double sigma,double rho,uint32_t count){
    reference=ref;exponent=eta;
    auto normal=CreateObject<NormalRandomVariable>();normal->SetStream(7);
    shadow.resize(count);shadow[0]=sigma*normal->GetValue(0,1);
    for(uint32_t i=1;i<count;++i)shadow[i]=rho*shadow[i-1]+sigma*std::sqrt(1-rho*rho)*normal->GetValue(0,1);
  }
 private:
  double DoCalcRxPower(double tx,Ptr<MobilityModel> a,Ptr<MobilityModel> b) const override {
    double d=std::max(1.,a->GetDistanceFrom(b));
    double out=tx-reference-10*exponent*std::log10(d);
    auto ia=a->GetObject<Node>()->GetId(),ib=b->GetObject<Node>()->GetId();
    bool target=(ia==0&&(ib==mainNode||ib==monitorNode))||(ib==0&&(ia==mainNode||ia==monitorNode));
    if(target&&!shadow.empty()){
      double t=Simulator::Now().GetSeconds();auto i=static_cast<uint32_t>(std::floor(t));
      i=std::min(i,static_cast<uint32_t>(shadow.size()-2));double f=t-std::floor(t);
      out+=shadow[i]*(1-f)+shadow[i+1]*f;
    }
    return out;
  }
  int64_t DoAssignStreams(int64_t) override{return 0;}
};
static void State(uint32_t role,Time start,Time duration,WifiPhyState state){
  double a=start.GetSeconds(),b=(start+duration).GetSeconds();
  if(b<=beginS||a>=endS||duration.IsZero())return;
  states<<role<<','<<start.GetNanoSeconds()<<','<<duration.GetNanoSeconds()<<','<<state<<'\n';
}
static void Radio(uint32_t role,Ptr<const Packet> p,uint16_t freq,WifiTxVector tx,MpduInfo ampdu,SignalNoiseDbm sn,uint16_t){
  double t=Simulator::Now().GetSeconds();if(t<beginS||t>=endS)return;
  auto mpdu=p->Copy();
  if(ampdu.type!=NORMAL_MPDU){
    AmpduSubframeHeader delimiter;mpdu->RemoveHeader(delimiter);
    NS_ABORT_MSG_IF(delimiter.GetLength()>mpdu->GetSize(),"Invalid A-MPDU delimiter length");
    mpdu=mpdu->CreateFragment(0,delimiter.GetLength());
  }
  WifiMacHeader h;mpdu->PeekHeader(h);
  if(!(h.IsData()||h.IsBeacon())||h.GetAddr2()!=apAddress)return;
  const bool main=h.IsData()&&h.GetAddr1()==mainAddress;
  frames<<role<<','<<Simulator::Now().GetNanoSeconds()<<','<<(h.IsBeacon()?"beacon":"data")<<','<<main<<','<<mpdu->GetSize()<<','
        <<sn.signal<<','<<sn.noise<<','<<freq<<','<<unsigned(tx.GetNss())<<','<<h.IsRetry()<<'\n';
}
static void Send(uint32_t flow){
  if(Simulator::Now()>=Seconds(endS))return;
  auto p=Create<Packet>(payload);int n=senders.at(flow)->Send(p);
  double t=Simulator::Now().GetSeconds();
  if(t>=beginS){if(n==static_cast<int>(payload))++sends.at(flow);else ++socketFails.at(flow);}
  Simulator::Schedule(Seconds(payload*8./(offeredMbps*1e6)),&Send,flow);
}
static void Receive(uint32_t flow,Ptr<Socket> s){
  Ptr<Packet> p;
  while((p=s->Recv())){
    const auto t=Simulator::Now().GetSeconds();
    if(t<beginS||t>=endS)continue;
    ++receives.at(flow);receivedBytes.at(flow)+=p->GetSize();
    apps<<Simulator::Now().GetNanoSeconds()<<','<<flow<<','<<p->GetSize()<<'\n';
  }
}
static void Cache(){NeighborCacheHelper c;c.PopulateNeighborCache();}
struct Point{double t,x,y;};
int main(int argc,char** argv){
  uint32_t run=20001,nClients=1;std::string trajectory="",mode="replay";double ref=40,eta=3,sigma=0,rho=.9,far=10000;
  bool noTraffic=false,pcap=false;double duration=12;
  CommandLine cmd(__FILE__);
  cmd.AddValue("run","RNG run",run);cmd.AddValue("clients","Associated saturated clients",nClients);
  cmd.AddValue("trajectory","Time,x,y CSV",trajectory);cmd.AddValue("mode","replay or static",mode);
  cmd.AddValue("referenceLoss","Effective reference loss dB",ref);cmd.AddValue("exponent","Effective pathloss exponent",eta);
  cmd.AddValue("sigma","Per-second shadow std dB",sigma);cmd.AddValue("rho","AR1 residual correlation",rho);
  cmd.AddValue("distance","Static distance m",far);cmd.AddValue("noTraffic","Engineering beacon-only",noTraffic);
  cmd.AddValue("duration","Measurement seconds",duration);cmd.AddValue("pcap","Engineering PCAP",pcap);
  cmd.AddValue("offeredMbps","Saturating per-client UDP workload",offeredMbps);cmd.Parse(argc,argv);
  NS_ABORT_MSG_IF(nClients<1||nClients>3,"Invalid client count");
  beginS=6;endS=beginS+duration;mainNode=1;monitorNode=nClients+1;
  RngSeedManager::SetSeed(6091420);RngSeedManager::SetRun(run);
  Config::SetDefault("ns3::WifiMacQueue::MaxSize",QueueSizeValue(QueueSize("512p")));
  Config::SetDefault("ns3::WifiMacQueue::MaxDelay",TimeValue(Seconds(.5)));
  Config::SetDefault("ns3::WifiMac::FrameRetryLimit",UintegerValue(7));
  Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",UintegerValue(65535));
  Config::SetDefault("ns3::WifiRemoteStationManager::FragmentationThreshold",UintegerValue(65535));
  NodeContainer nodes;nodes.Create(nClients+2);
  MobilityHelper mobile;mobile.SetMobilityModel("ns3::WaypointMobilityModel","InitialPositionIsWaypoint",BooleanValue(true));mobile.Install(nodes);
  nodes.Get(0)->GetObject<WaypointMobilityModel>()->SetPosition(Vector(0,0,0));
  for(uint32_t i=2;i<=nClients;++i)nodes.Get(i)->GetObject<WaypointMobilityModel>()->SetPosition(Vector(2,double(i-2)*.5,0));
  std::vector<Point> points;
  if(mode=="replay"){
    std::ifstream in(trajectory);NS_ABORT_MSG_IF(!in,"Missing trajectory");std::string line;std::getline(in,line);
    while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream ss(line);Point q;ss>>q.t>>q.x>>q.y;NS_ABORT_MSG_IF(!ss,"Invalid trajectory");points.push_back(q);}
    NS_ABORT_MSG_IF(points.empty()||points.front().t!=2||points.back().t<endS,"Insufficient path");
  }else points={{2,far,0},{endS+1,far,0}};
  for(auto i:{mainNode,monitorNode}){
    auto mob=nodes.Get(i)->GetObject<WaypointMobilityModel>();mob->SetPosition(Vector(5,0,0));
    // Complete association before beginning the public path and four-second history.
    mob->AddWaypoint(Waypoint(Seconds(1.999999),Vector(5,0,0)));
    for(const auto& q:points)mob->AddWaypoint(Waypoint(Seconds(q.t),Vector(q.x,q.y,0)));
  }
  auto loss=CreateObject<PilotLoss>();loss->Configure(ref,eta,sigma,rho,static_cast<uint32_t>(endS)+3);
  auto channel=CreateObject<YansWifiChannel>();channel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());channel->SetPropagationLossModel(loss);
  YansWifiPhyHelper phy;phy.SetChannel(channel);phy.Set("ChannelSettings",StringValue("{6,20,BAND_2_4GHZ,0}"));
  phy.Set("TxPowerStart",DoubleValue(20));phy.Set("TxPowerEnd",DoubleValue(20));phy.Set("RxNoiseFigure",DoubleValue(7));
  phy.Set("Antennas",UintegerValue(2));phy.Set("MaxSupportedTxSpatialStreams",UintegerValue(2));phy.Set("MaxSupportedRxSpatialStreams",UintegerValue(2));
  phy.SetErrorRateModel("ns3::NistErrorRateModel");
  WifiHelper wifi;wifi.SetStandard(WIFI_STANDARD_80211n);wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");
  wifi.ConfigHtOptions("ShortGuardIntervalSupported",BooleanValue(false));
  WifiMacHelper mac;Ssid ssid("p20-infrastructure");NetDeviceContainer all;
  mac.SetType("ns3::ApWifiMac","Ssid",SsidValue(ssid),"EnableBeaconJitter",BooleanValue(false),"BE_MaxAmpduSize",UintegerValue(65535));
  all.Add(wifi.Install(phy,mac,nodes.Get(0)));
  mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"ActiveProbing",BooleanValue(false),"BE_MaxAmpduSize",UintegerValue(65535));
  for(uint32_t i=1;i<=nClients;++i)all.Add(wifi.Install(phy,mac,nodes.Get(i)));
  mac.SetType("ns3::AdhocWifiMac","Ssid",SsidValue(Ssid("p20-passive")),"QosSupported",BooleanValue(true));
  all.Add(wifi.Install(phy,mac,nodes.Get(monitorNode)));
  auto monitor=DynamicCast<WifiNetDevice>(all.Get(monitorNode));monitor->GetMac()->SetPromisc();
  const auto streams=WifiHelper::AssignStreams(all,100);
  auto ap=DynamicCast<WifiNetDevice>(all.Get(0));auto main=DynamicCast<WifiNetDevice>(all.Get(1));
  apAddress=Mac48Address::ConvertFrom(ap->GetAddress());mainAddress=Mac48Address::ConvertFrom(main->GetAddress());
  InternetStackHelper net;NodeContainer active;
  for(uint32_t i=0;i<=nClients;++i)active.Add(nodes.Get(i));net.Install(active);
  NetDeviceContainer activeDevices;for(uint32_t i=0;i<=nClients;++i)activeDevices.Add(all.Get(i));
  Ipv4AddressHelper ip;ip.SetBase("10.20.1.0","255.255.255.0");auto addresses=ip.Assign(activeDevices);
  sends.resize(nClients);socketFails.resize(nClients);receives.resize(nClients);receivedBytes.resize(nClients);
  for(uint32_t i=0;i<nClients;++i){
    auto sink=Socket::CreateSocket(nodes.Get(i+1),UdpSocketFactory::GetTypeId());
    NS_ABORT_MSG_IF(sink->Bind(InetSocketAddress(Ipv4Address::GetAny(),9000+i))<0,"Bind failed");sink->SetRecvCallback(MakeBoundCallback(&Receive,i));sinks.push_back(sink);
    auto tx=Socket::CreateSocket(nodes.Get(0),UdpSocketFactory::GetTypeId());
    NS_ABORT_MSG_IF(tx->Connect(InetSocketAddress(addresses.GetAddress(i+1),9000+i))<0,"Connect failed");senders.push_back(tx);
    if(!noTraffic)Simulator::Schedule(Seconds(1.5+double(i)*.00001),&Send,i);
  }
  states.open("phy_states.csv");frames.open("rx_frames.csv");apps.open("udp_receives.csv");
  states<<"role,start_ns,duration_ns,state\n";frames<<"role,time_ns,kind,to_main,frame_bytes,rssi_dbm,noise_dbm,frequency_mhz,nss,retry\n"<<std::setprecision(17);
  apps<<"time_ns,flow,payload_bytes\n";
  std::array<Ptr<WifiNetDevice>,3> observed={ap,main,monitor};
  for(uint32_t i=0;i<3;++i){
    NS_ABORT_MSG_IF(!observed[i]->GetPhy()->GetState()->TraceConnectWithoutContext("State",MakeBoundCallback(&State,i)),"Missing state trace");
    if(i)NS_ABORT_MSG_IF(!observed[i]->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx",MakeBoundCallback(&Radio,i)),"Missing radio trace");
    // Flush a final idle interval after the analysis horizon; never changes measured time.
    Simulator::Schedule(Seconds(endS+.25),&WifiPhy::SetOffMode,observed[i]->GetPhy());
  }
  if(pcap)phy.EnablePcap("engineering-monitor",all.Get(monitorNode),true);
  Simulator::Schedule(Seconds(1.2),&Cache);
  Simulator::Stop(Seconds(endS+.5));Simulator::Run();states.close();frames.close();apps.close();
  std::ofstream summary("traffic_summary.csv");summary<<"flow,socket_accepted,socket_failed,udp_received,udp_bytes\n";
  for(uint32_t i=0;i<nClients;++i)summary<<i<<','<<sends[i]<<','<<socketFails[i]<<','<<receives[i]<<','<<receivedBytes[i]<<'\n';
  std::ofstream meta("runtime.csv");meta<<"run,clients,streams,begin_s,end_s,reference_loss,exponent,sigma,rho,offered_mbps\n"<<std::setprecision(17)
    <<run<<','<<nClients<<','<<streams<<','<<beginS<<','<<endS<<','<<ref<<','<<eta<<','<<sigma<<','<<rho<<','<<offeredMbps<<'\n';
  std::cout<<"completed run="<<run<<" clients="<<nClients<<" primary_udp_packets="<<receives[0]<<"\n";
  Simulator::Destroy();return 0;
}
