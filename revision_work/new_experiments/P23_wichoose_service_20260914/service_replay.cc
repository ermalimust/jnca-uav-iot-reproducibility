// P23 conditional uplink service replay, independently authored. GPL-2.0-only.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/neighbor-cache-helper.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <vector>
#include <algorithm>
using namespace ns3;
constexpr uint32_t payload=1440;
struct Row {double rssi=-35;uint32_t packets=0;};
static std::vector<Row> input;
static std::vector<uint64_t> accepted,failed,received,rxBytes,radioN;
static std::vector<double> radioSum;
static Ptr<Socket> txSocket;
static std::ofstream events;
static uint64_t nextId=0,duplicates=0,badSize=0,unsent=0;
static std::unordered_set<uint64_t> seen;
static double beginS=6,endS=26;
class TracePower:public PropagationLossModel {
 public:
  static TypeId GetTypeId(){static TypeId id=TypeId("ns3::P23TracePower").SetParent<PropagationLossModel>().AddConstructor<TracePower>();return id;}
 private:
  double DoCalcRxPower(double,Ptr<MobilityModel>,Ptr<MobilityModel>) const override {
    auto i=std::min(size_t(Simulator::Now().GetSeconds()),input.size()-1);return input[i].rssi;
  }
  int64_t DoAssignStreams(int64_t) override{return 0;}
};
class IdHeader:public Header {
 public:
  uint64_t id=0;
  static TypeId GetTypeId(){static TypeId t=TypeId("ns3::P23IdHeader").SetParent<Header>().AddConstructor<IdHeader>();return t;}
  TypeId GetInstanceTypeId() const override{return GetTypeId();}
  uint32_t GetSerializedSize()const override{return 8;}
  void Serialize(Buffer::Iterator i)const override{i.WriteHtonU32(id>>32);i.WriteHtonU32(id&0xffffffff);}
  uint32_t Deserialize(Buffer::Iterator i)override{id=uint64_t(i.ReadNtohU32())<<32;id|=i.ReadNtohU32();return 8;}
  void Print(std::ostream& out)const override{out<<id;}
};
static void Send(){
 auto sec=size_t(Simulator::Now().GetSeconds());IdHeader h;h.id=nextId++;
 auto p=Create<Packet>(payload-8);p->AddHeader(h);auto n=txSocket->Send(p);
 if(n==int(payload))accepted[sec]++;else failed[sec]++;
}
static void StartSecond(uint32_t sec){
 auto count=input[sec].packets;
 for(uint32_t j=0;j<count;++j)Simulator::Schedule(Seconds((j+.5)/double(count)),&Send);
}
static void Receive(Ptr<Socket> s){
 Ptr<Packet> p;
 while((p=s->Recv())){
  auto bytes=p->GetSize();if(bytes!=payload){badSize++;continue;}
  IdHeader h;p->RemoveHeader(h);if(h.id>=nextId)unsent++;
  if(!seen.insert(h.id).second)duplicates++;
  auto sec=std::min(size_t(Simulator::Now().GetSeconds()),received.size()-1);
  received[sec]++;rxBytes[sec]+=bytes;
  events<<Simulator::Now().GetNanoSeconds()<<','<<h.id<<','<<bytes<<'\n';
 }
}
static void Radio(Ptr<const Packet> p,uint16_t,WifiTxVector,MpduInfo ampdu,SignalNoiseDbm sn,uint16_t){
 auto c=p->Copy();if(ampdu.type!=NORMAL_MPDU){AmpduSubframeHeader h;c->RemoveHeader(h);c=c->CreateFragment(0,h.GetLength());}
 WifiMacHeader h;c->PeekHeader(h);if(!h.IsData())return;
 auto sec=std::min(size_t(Simulator::Now().GetSeconds()),radioN.size()-1);
 radioN[sec]++;radioSum[sec]+=sn.signal;
}
static void Cache(){NeighborCacheHelper c;c.PopulateNeighborCache();}
int main(int argc,char** argv){
 std::string path="";uint32_t run=23001,duration=20;double rssi=-35,mbps=10;bool step=false;
 CommandLine cmd(__FILE__);cmd.AddValue("input","sim_second,rssi_dbm,packets CSV",path);
 cmd.AddValue("run","RNG run",run);cmd.AddValue("duration","Measurement seconds",duration);
 cmd.AddValue("rssi","Engineering effective RSSI",rssi);cmd.AddValue("mbps","Engineering offered Mbps",mbps);
 cmd.AddValue("step","Engineering workload step",step);cmd.Parse(argc,argv);
 endS=beginS+duration;input.resize(size_t(endS)+2);
 for(auto& x:input)x=Row{-35,0};
 if(path.empty()){
  for(size_t i=2;i<size_t(endS);++i)input[i]=Row{rssi,uint32_t(mbps*1e6/(payload*8))};
  if(step)for(size_t i=2;i<size_t(endS);++i)input[i].packets= (i%2==0?100:1000);
 }else{
  std::ifstream in(path);NS_ABORT_MSG_IF(!in,"Missing input");std::string line;std::getline(in,line);uint32_t count=0;
  while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream ss(line);uint32_t t,n;double s;ss>>t>>s>>n;
   NS_ABORT_MSG_IF(!ss||t!=count+2||t>=endS||!std::isfinite(s),"Invalid input");input[t]=Row{s,n};count++;}
  NS_ABORT_MSG_IF(count!=uint32_t(endS)-2,"Incomplete input");
 }
 input[size_t(endS)]=input[size_t(endS)-1];input.back()=input[size_t(endS)-1];
 auto size=input.size();accepted.resize(size);failed.resize(size);received.resize(size);rxBytes.resize(size);radioN.resize(size);radioSum.resize(size);
 RngSeedManager::SetSeed(6091423);RngSeedManager::SetRun(run);
 Config::SetDefault("ns3::WifiMacQueue::MaxSize",QueueSizeValue(QueueSize("512p")));
 Config::SetDefault("ns3::WifiMacQueue::MaxDelay",TimeValue(Seconds(.5)));
 Config::SetDefault("ns3::WifiMac::FrameRetryLimit",UintegerValue(7));
 Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",UintegerValue(65535));
 Config::SetDefault("ns3::WifiRemoteStationManager::FragmentationThreshold",UintegerValue(65535));
 NodeContainer nodes;nodes.Create(2);MobilityHelper mob;mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");mob.Install(nodes);
 nodes.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(1,0,0));
 auto channel=CreateObject<YansWifiChannel>();channel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());channel->SetPropagationLossModel(CreateObject<TracePower>());
 YansWifiPhyHelper phy;phy.SetChannel(channel);phy.Set("ChannelSettings",StringValue("{6,20,BAND_2_4GHZ,0}"));
 phy.Set("TxPowerStart",DoubleValue(20));phy.Set("TxPowerEnd",DoubleValue(20));phy.Set("RxNoiseFigure",DoubleValue(7));
 phy.Set("Antennas",UintegerValue(2));phy.Set("MaxSupportedTxSpatialStreams",UintegerValue(2));phy.Set("MaxSupportedRxSpatialStreams",UintegerValue(2));phy.SetErrorRateModel("ns3::NistErrorRateModel");
 WifiHelper wifi;wifi.SetStandard(WIFI_STANDARD_80211n);wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");wifi.ConfigHtOptions("ShortGuardIntervalSupported",BooleanValue(false));
 WifiMacHelper mac;Ssid ssid("p23-service");NetDeviceContainer all;
 mac.SetType("ns3::ApWifiMac","Ssid",SsidValue(ssid),"EnableBeaconJitter",BooleanValue(false),"BE_MaxAmpduSize",UintegerValue(65535));all.Add(wifi.Install(phy,mac,nodes.Get(0)));
 mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"ActiveProbing",BooleanValue(false),"BE_MaxAmpduSize",UintegerValue(65535));all.Add(wifi.Install(phy,mac,nodes.Get(1)));
 WifiHelper::AssignStreams(all,100);InternetStackHelper stack;stack.Install(nodes);Ipv4AddressHelper ip;ip.SetBase("10.23.1.0","255.255.255.0");auto addresses=ip.Assign(all);
 auto sink=Socket::CreateSocket(nodes.Get(0),UdpSocketFactory::GetTypeId());NS_ABORT_MSG_IF(sink->Bind(InetSocketAddress(Ipv4Address::GetAny(),9000))<0,"Bind");sink->SetRecvCallback(MakeCallback(&Receive));
 txSocket=Socket::CreateSocket(nodes.Get(1),UdpSocketFactory::GetTypeId());NS_ABORT_MSG_IF(txSocket->Connect(InetSocketAddress(addresses.GetAddress(0),9000))<0,"Connect");
 auto ap=DynamicCast<WifiNetDevice>(all.Get(0));NS_ABORT_MSG_IF(!ap->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx",MakeCallback(&Radio)),"Trace");
 events.open("receives.csv");events<<"time_ns,packet_id,payload_bytes\n";
 for(uint32_t sec=2;sec<uint32_t(endS);++sec)Simulator::Schedule(Seconds(sec),&StartSecond,sec);
 Simulator::Schedule(Seconds(1.2),&Cache);Simulator::Stop(Seconds(endS+1));Simulator::Run();events.close();
 std::ofstream out("seconds.csv");out<<"sim_second,measured,input_rssi_dbm,scheduled_packets,socket_accepted,socket_failed,rx_packets,rx_bytes,radio_frames,radio_rssi_mean\n"<<std::setprecision(17);
 for(size_t i=0;i<size;++i)out<<i<<','<<(i>=size_t(beginS)&&i<size_t(endS))<<','<<input[i].rssi<<','<<(i<size_t(endS)?input[i].packets:0)<<','<<accepted[i]<<','<<failed[i]<<','<<received[i]<<','<<rxBytes[i]<<','<<radioN[i]<<','<<(radioN[i]?radioSum[i]/radioN[i]:0)<<'\n';
 std::ofstream check("integrity.json");check<<"{\"duplicates\":"<<duplicates<<",\"bad_size\":"<<badSize<<",\"unsent_id\":"<<unsent<<",\"generated_packets\":"<<nextId<<",\"unique_received\":"<<seen.size()<<"}";
 std::cout<<"run="<<run<<" generated="<<nextId<<" unique_rx="<<seen.size()<<"\n";Simulator::Destroy();return 0;
}
