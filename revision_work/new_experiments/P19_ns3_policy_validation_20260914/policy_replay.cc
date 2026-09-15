// P19 independent observable-prefix policy replay; derived from frozen P18. SPDX-License-Identifier: GPL-2.0-only
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/neighbor-cache-helper.h"
#include <deque>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace ns3;
class LedgerTag : public Tag {
public:
  uint32_t id=0;
  static TypeId GetTypeId(){static TypeId t=TypeId("ns3::P19LedgerTag").SetParent<Tag>().AddConstructor<LedgerTag>();return t;}
  TypeId GetInstanceTypeId() const override{return GetTypeId();}
  uint32_t GetSerializedSize() const override{return 4;}
  void Serialize(TagBuffer b) const override{b.WriteU32(id);}
  void Deserialize(TagBuffer b) override{id=b.ReadU32();}
  void Print(std::ostream& os) const override{os<<id;}
};
struct Row {
  uint32_t id,flow,bytes; int64_t scheduled_ns;
  bool offered=false; int64_t offer_ns=-1,send_ns=-1,receive_ns=-1,shaper_drop_ns=-1,socket_fail_ns=-1;
  uint32_t duplicates=0,phy_attempts=0,mac_acks=0,mac_drops=0,tid_mask=0;
  int64_t first_phy_ns=-1,last_phy_ns=-1,last_mac_drop_ns=-1,first_ack_ns=-1; uint64_t drop_reason_mask=0;
};
static std::vector<Row> rows;
static std::vector<Ptr<Socket>> txSockets,sinks;
static std::deque<uint32_t> videoQueue;
static Ptr<WifiNetDevice> primary[2];
static bool drainScheduled=false;
static double videoCap=0;
static Time nextVideo=Seconds(0);
static std::string action="Observe",prefixPath="prefix.csv",outputPath="packets.csv",knobPath="knobs.csv";
static uint32_t mFactor=0;
static double motionSpeed=3.5;
static bool probe=false;
struct RadioRow {int64_t time_ns; uint32_t packet_id; double signal_dbm,noise_dbm;};
static std::vector<RadioRow> controllerVideoRadio;
static void RadioRx(Ptr<const Packet> p,uint16_t,WifiTxVector,MpduInfo,SignalNoiseDbm sn,uint16_t){
  LedgerTag t;if(!p->PeekPacketTag(t)||rows.at(t.id).flow!=1)return;
  controllerVideoRadio.push_back({Simulator::Now().GetNanoSeconds(),t.id,sn.signal,sn.noise});
}
static void WriteRadio(){
  std::ofstream out("radio_prefix.csv");NS_ABORT_MSG_IF(!out,"Cannot write controller radio observations");
  out<<"time_ns,packet_id,signal_dbm,noise_dbm\n"<<std::setprecision(17);
  for(const auto& r:controllerVideoRadio)out<<r.time_ns<<','<<r.packet_id<<','<<r.signal_dbm<<','<<r.noise_dbm<<'\n';
}
static int64_t now(){return Simulator::Now().GetNanoSeconds();}

static void Send(uint32_t id){
  auto& r=rows.at(id);auto p=Create<Packet>(r.bytes);LedgerTag tag;tag.id=id;p->AddPacketTag(tag);
  int n=txSockets.at(r.flow)->Send(p);
  if(n==static_cast<int>(r.bytes))r.send_ns=now();else r.socket_fail_ns=now();
}
static void Drain(){
  drainScheduled=false;
  if(videoQueue.empty()||Simulator::Now()>=Seconds(23))return;
  auto id=videoQueue.front();videoQueue.pop_front();Send(id);
  auto dt=NanoSeconds(static_cast<int64_t>(rows[id].bytes*8.0/videoCap*1e9));nextVideo=Simulator::Now()+dt;
  if(!videoQueue.empty()){drainScheduled=true;Simulator::Schedule(dt,&Drain);}
}
static void Offer(uint32_t id){
  auto& r=rows.at(id);r.offered=true;r.offer_ns=now();
  if(r.flow!=1||videoCap==0){Send(id);return;}
  if(videoQueue.size()>=128){r.shaper_drop_ns=now();return;}
  videoQueue.push_back(id);
  if(!drainScheduled){drainScheduled=true;Simulator::Schedule(std::max(Seconds(0),nextVideo-Simulator::Now()),&Drain);}
}
static void Receive(Ptr<Socket> s){
  Ptr<Packet> p;
  while((p=s->Recv())){
    LedgerTag t;NS_ABORT_MSG_IF(!p->PeekPacketTag(t),"Received application packet without ID tag");
    auto& r=rows.at(t.id);NS_ABORT_MSG_IF(p->GetSize()!=r.bytes,"Payload size changed");
    if(r.receive_ns<0)r.receive_ns=now();else ++r.duplicates;
  }
}
static void PhyTx(Ptr<const Packet> p,double /*power*/){
  LedgerTag t;if(!p->PeekPacketTag(t))return;
  auto& r=rows.at(t.id);++r.phy_attempts;if(r.first_phy_ns<0)r.first_phy_ns=now();r.last_phy_ns=now();
  WifiMacHeader h;p->PeekHeader(h);if(h.IsQosData())r.tid_mask|=1u<<h.GetQosTid();
}
static void Acked(Ptr<const WifiMpdu> mpdu){
  LedgerTag t;if(!mpdu->GetPacket()->PeekPacketTag(t))return;
  auto& r=rows.at(t.id);++r.mac_acks;if(r.first_ack_ns<0)r.first_ack_ns=now();if(mpdu->GetHeader().IsQosData())r.tid_mask|=1u<<mpdu->GetHeader().GetQosTid();
}
static void Dropped(WifiMacDropReason reason,Ptr<const WifiMpdu> mpdu){
  LedgerTag t;if(!mpdu->GetPacket()->PeekPacketTag(t))return;
  auto& r=rows.at(t.id);++r.mac_drops;r.last_mac_drop_ns=now();r.drop_reason_mask|=1ull<<static_cast<unsigned>(reason);
  if(mpdu->GetHeader().IsQosData())r.tid_mask|=1u<<mpdu->GetHeader().GetQosTid();
}
static void WriteRows(std::string path){
  std::ofstream out(path);NS_ABORT_MSG_IF(!out,"Cannot write packet ledger");
  out<<"packet_id,flow,payload_bytes,offer_ns,send_ns,receive_ns,shaper_drop_ns,socket_fail_ns,duplicates,phy_attempts,first_phy_ns,last_phy_ns,mac_acks,mac_drops,last_mac_drop_ns,drop_reason_mask,tid_mask,first_ack_ns\n";
  for(const auto& r:rows){if(!r.offered)continue;
    out<<r.id<<','<<r.flow<<','<<r.bytes<<','<<r.offer_ns<<','<<r.send_ns<<','<<r.receive_ns<<','<<r.shaper_drop_ns<<','<<r.socket_fail_ns<<','<<r.duplicates<<','<<r.phy_attempts<<','<<r.first_phy_ns<<','<<r.last_phy_ns<<','<<r.mac_acks<<','<<r.mac_drops<<','<<r.last_mac_drop_ns<<','<<r.drop_reason_mask<<','<<r.tid_mask<<','<<r.first_ack_ns<<'\n';
  }
}
static void Knobs(std::string phase){
  std::ofstream out(knobPath,phase=="before"?std::ios::out:std::ios::app);
  if(phase=="before")out<<"phase,time_ns,node,mode,cwmin_be,cwmax_be,aifsn_be,c2_tos,video_cap_bps,x_m,tx_power_dbm\n";
  for(int i=0;i<2;++i){StringValue mode;primary[i]->GetRemoteStationManager()->GetAttribute("DataMode",mode);auto be=primary[i]->GetMac()->GetQosTxop(AC_BE);
    out<<phase<<','<<now()<<','<<i<<','<<mode.Get()<<','<<be->GetMinCw(0)<<','<<be->GetMaxCw(0)<<','<<unsigned(be->GetAifsn(0))<<','<<unsigned(txSockets[0]->GetIpTos())<<','<<std::fixed<<std::setprecision(6)<<videoCap<<','<<primary[i]->GetNode()->GetObject<MobilityModel>()->GetPosition().x<<','<<primary[i]->GetPhy()->GetTxPowerStart()<<'\n';
  }
}
static void Activate(){
  if(action=="WiFiRelief"){
    auto be=primary[1]->GetMac()->GetQosTxop(AC_BE);be->SetMinCw(63);be->SetMaxCw(1023);be->SetAifsn(7);
  }else if(action=="LinkAdapt"){
    for(auto d:primary)d->GetRemoteStationManager()->SetAttribute("DataMode",StringValue("OfdmRate6Mbps"));
  }else if(action=="VideoShape")videoCap=4000000;
  else if(action=="FallbackProtect"){videoCap=1000000;txSockets[0]->SetIpTos(0xc0);}
  else NS_ABORT_MSG_IF(action!="Observe","Unknown action");
  nextVideo=Simulator::Now();Knobs("after");
}
static void Cache(){NeighborCacheHelper n;n.PopulateNeighborCache();}

int main(int argc,char** argv){
  std::string input="offers.csv";uint32_t run=9001;bool zero=false,pcap=false;double distance=15;
  CommandLine cmd(__FILE__);cmd.AddValue("action","One registered action",action);cmd.AddValue("input","Frozen offer schedule",input);
  cmd.AddValue("output","Final packet ledger",outputPath);cmd.AddValue("prefix","Pre-command snapshot",prefixPath);cmd.AddValue("knobs","Parameter readback",knobPath);
  cmd.AddValue("run","RNG run",run);cmd.AddValue("moving","M factor",mFactor);cmd.AddValue("zero","Engineering-only no traffic",zero);cmd.AddValue("pcap","Engineering PCAP",pcap);cmd.AddValue("distance","Engineering-only initial range",distance);cmd.AddValue("speed","Motion speed in m/s",motionSpeed);cmd.AddValue("probe","Stop after prefix before any intervention",probe);cmd.Parse(argc,argv);
  NS_ABORT_MSG_IF(mFactor>1,"Invalid mobility factor");RngSeedManager::SetSeed(6091419);RngSeedManager::SetRun(run);
  Config::SetDefault("ns3::WifiMacQueue::MaxSize",QueueSizeValue(QueueSize("128p")));
  Config::SetDefault("ns3::WifiMacQueue::MaxDelay",TimeValue(Seconds(.5)));
  Config::SetDefault("ns3::WifiMac::FrameRetryLimit",UintegerValue(7));
  Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",UintegerValue(65535));
  Config::SetDefault("ns3::WifiRemoteStationManager::FragmentationThreshold",UintegerValue(65535));
  NodeContainer nodes;nodes.Create(4);MobilityHelper mobility;mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");mobility.Install(nodes);
  std::vector<Vector> positions={{0,0,0},{distance,0,0},{0,20,0},{15,20,0}};
  for(int i=0;i<4;++i)nodes.Get(i)->GetObject<ConstantVelocityMobilityModel>()->SetPosition(positions[i]);
  if(mFactor){auto model=nodes.Get(1)->GetObject<ConstantVelocityMobilityModel>();Simulator::Schedule(Seconds(1),&ConstantVelocityMobilityModel::SetVelocity,model,Vector(motionSpeed,0,0));Simulator::Schedule(Seconds(21),&ConstantVelocityMobilityModel::SetVelocity,model,Vector(0,0,0));}
  YansWifiChannelHelper channel;channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel","Exponent",DoubleValue(3),"ReferenceDistance",DoubleValue(1),"ReferenceLoss",DoubleValue(46.6777));
  YansWifiPhyHelper phy;phy.SetChannel(channel.Create());phy.Set("ChannelSettings",StringValue("{36,20,BAND_5GHZ,0}"));
  phy.Set("TxPowerStart",DoubleValue(16));phy.Set("TxPowerEnd",DoubleValue(16));phy.Set("RxNoiseFigure",DoubleValue(7));phy.SetErrorRateModel("ns3::NistErrorRateModel");
  WifiHelper wifi;wifi.SetStandard(WIFI_STANDARD_80211a);wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue("OfdmRate24Mbps"),"ControlMode",StringValue("OfdmRate6Mbps"));
  WifiMacHelper mac;NetDeviceContainer devices;
  for(int i=0;i<4;++i){mac.SetType("ns3::AdhocWifiMac","QosSupported",BooleanValue(true),"Ssid",SsidValue(Ssid(i<2?"p18-primary":"p18-external")));devices.Add(wifi.Install(phy,mac,nodes.Get(i)));}
  const auto usedStreams=WifiHelper::AssignStreams(devices,100);
  for(int i=0;i<2;++i)primary[i]=DynamicCast<WifiNetDevice>(devices.Get(i));
  InternetStackHelper internet;internet.Install(nodes);Ipv4AddressHelper ip;ip.SetBase("10.1.1.0","255.255.255.0");
  NetDeviceContainer d1;d1.Add(devices.Get(0));d1.Add(devices.Get(1));auto i1=ip.Assign(d1);
  ip.SetBase("10.2.1.0","255.255.255.0");NetDeviceContainer d2;d2.Add(devices.Get(2));d2.Add(devices.Get(3));auto i2=ip.Assign(d2);
  std::vector<int> src={0,1,2},dst={1,0,3};std::vector<Ipv4Address> dest={i1.GetAddress(1),i1.GetAddress(0),i2.GetAddress(1)};
  for(int f=0;f<3;++f){auto sink=Socket::CreateSocket(nodes.Get(dst[f]),UdpSocketFactory::GetTypeId());NS_ABORT_MSG_IF(sink->Bind(InetSocketAddress(Ipv4Address::GetAny(),9000+f))<0,"Bind failed");sink->SetRecvCallback(MakeCallback(&Receive));sinks.push_back(sink);
    auto sock=Socket::CreateSocket(nodes.Get(src[f]),UdpSocketFactory::GetTypeId());sock->SetIpTos(0);NS_ABORT_MSG_IF(sock->Connect(InetSocketAddress(dest[f],9000+f))<0,"Connect failed");txSockets.push_back(sock);}
  for(int i=0;i<4;++i){auto d=DynamicCast<WifiNetDevice>(devices.Get(i));NS_ABORT_MSG_IF(!d->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeCallback(&PhyTx)),"Missing PHY trace");NS_ABORT_MSG_IF(!d->GetMac()->TraceConnectWithoutContext("AckedMpdu",MakeCallback(&Acked)),"Missing ACK trace");NS_ABORT_MSG_IF(!d->GetMac()->TraceConnectWithoutContext("DroppedMpdu",MakeCallback(&Dropped)),"Missing drop trace");}
  NS_ABORT_MSG_IF(!primary[0]->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx",MakeCallback(&RadioRx)),"Missing controller RSSI trace");
  if(pcap)phy.EnablePcapAll("engineering",true);
  std::ifstream in(input);NS_ABORT_MSG_IF(!in,"Missing frozen offers");std::string line;std::getline(in,line);
  while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream ss(line);uint32_t id,flow,bytes;int64_t rel;
    ss>>id>>flow>>rel>>bytes;NS_ABORT_MSG_IF(!ss||id!=rows.size()||flow>2||bytes==0||rel<0||rel>=20000000,"Invalid offer row");
    Row r;r.id=id;r.flow=flow;r.bytes=bytes;r.scheduled_ns=1000000000ll+rel*1000;rows.push_back(r);if(!zero)Simulator::Schedule(NanoSeconds(r.scheduled_ns),&Offer,id);}
  Simulator::Schedule(Seconds(.5),&Cache);
  Simulator::Schedule(NanoSeconds(10999999999ll),&WriteRows,prefixPath);
  Simulator::Schedule(NanoSeconds(10999999999ll),&WriteRadio);
  Simulator::Schedule(Seconds(11),&Knobs,std::string("before"));
  Simulator::Schedule(Seconds(11.001),&Activate);
  Simulator::Stop(probe?NanoSeconds(10999999999ll):Seconds(23));Simulator::Run();WriteRows(outputPath);
  std::cout<<"completed action="<<action<<" run="<<run<<" moving="<<mFactor<<" scheduled="<<rows.size()<<" streams="<<usedStreams<<" shaper_pending="<<videoQueue.size()<<"\n";
  Simulator::Destroy();return 0;
}
