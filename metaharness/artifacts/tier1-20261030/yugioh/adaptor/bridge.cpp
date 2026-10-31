// SPDX-License-Identifier: AGPL-3.0-or-later
// Stable flat ABI around the pinned EDOPro 11.0 API.
#include "ocgapi.h"
#include "ocgapi_constants.h"
#include <array>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>
struct Entry { OCG_CardData data{}; std::array<uint16_t,5> sets{}; };
struct Handle {
    OCG_Duel duel{};
    std::unordered_map<uint32_t,Entry> cards;
    std::string scripts, error;
};
static void reader(void* p, uint32_t code, OCG_CardData* out) {
    auto& h=*static_cast<Handle*>(p);
    auto it=h.cards.find(code);
    if(it==h.cards.end()) { h.error="Missing card metadata: "+std::to_string(code); *out={}; return; }
    *out=it->second.data;
    out->setcodes=it->second.sets.data();
}
static int script(void* p, OCG_Duel d, const char* name) {
    auto& h=*static_cast<Handle*>(p);
    std::string n(name);
    if(n=="c0.lua") return 0; // engine dummy card has no script
    if(n.find("..")!=std::string::npos || n.find('/')!=std::string::npos) {h.error="Unsafe script path";return 0;}
    for(const auto& prefix: {std::string("/"), std::string("/official/"), std::string("/unofficial/")}) {
        std::ifstream f(h.scripts+prefix+n, std::ios::binary);
        if(f) {
            std::string s((std::istreambuf_iterator<char>(f)),{});
            int ok=OCG_LoadScript(d,s.data(),s.size(),name);
            if(!ok) h.error="Failed loading script: "+n;
            return ok;
        }
    }
    // The engine probes optional scripts for vanilla Normal Monsters, too.
    if(n.size()>5 && n[0]=='c') {
        try {auto id=std::stoul(n.substr(1)); auto it=h.cards.find(id);
            if(it!=h.cards.end() && (it->second.data.type & TYPE_NORMAL)) return 0;
        } catch(...) {}
    }
    h.error="Missing script: "+n;
    return 0;
}
static void logger(void* p,const char* msg,int type) {
    if(type==OCG_LOG_TYPE_ERROR) static_cast<Handle*>(p)->error=msg;
}
extern "C" {
void* ygo_create(const char* tsv,const char* scripts,const uint64_t* seed) {
    auto* h=new Handle(); h->scripts=scripts;
    std::ifstream f(tsv);
    uint64_t id,alias,sets,type,level,race,attr; int64_t atk,def;
    while(f>>id>>alias>>sets>>type>>level>>race>>attr>>atk>>def) {
        auto& e=h->cards[id]; auto& d=e.data;
        d.code=id;d.alias=alias;d.type=type;d.level=level&0xff;
        d.lscale=(level>>24)&0xff;d.rscale=(level>>16)&0xff;
        d.race=race;d.attribute=attr;d.attack=atk;
        d.defense=(type&TYPE_LINK)?0:def;d.link_marker=(type&TYPE_LINK)?def:0;
        for(int i=0;i<4;++i)e.sets[i]=(sets>>(16*i))&0xffff;
    }
    if(h->cards.empty()) {h->error="Empty metadata file";return h;}
    int major=0,minor=0; OCG_GetVersion(&major,&minor);
    if(major!=11 || minor!=0) {h->error="Core ABI version mismatch";return h;}
    OCG_DuelOptions o{};std::memcpy(o.seed,seed,32);o.flags=DUEL_MODE_MR5;
    o.team1={8000,5,1};o.team2={8000,5,1};
    o.cardReader=reader;o.payload1=h;o.scriptReader=script;o.payload2=h;
    o.logHandler=logger;o.payload3=h;o.enableUnsafeLibraries=0;
    int result=OCG_CreateDuel(&h->duel,&o);
    if(result) {h->error="CreateDuel status "+std::to_string(result);return h;}
    script(h,h->duel,"constant.lua");script(h,h->duel,"utility.lua");
    return h;
}
const char* ygo_error(Handle* h) {return h->error.c_str();}
void ygo_destroy(Handle* h) {if(h){OCG_DestroyDuel(h->duel);delete h;}}
void ygo_card(Handle* h,uint32_t code,uint8_t player,uint32_t loc,uint32_t seq,uint32_t pos) {
    OCG_NewCardInfo i{};i.team=player;i.con=player;i.code=code;i.loc=loc;i.seq=seq;i.pos=pos;
    OCG_DuelNewCard(h->duel,&i);
}
void ygo_start(Handle* h){OCG_StartDuel(h->duel);}
int ygo_process(Handle* h){return OCG_DuelProcess(h->duel);}
const void* ygo_message(Handle* h,uint32_t* len){return OCG_DuelGetMessage(h->duel,len);}
void ygo_response(Handle* h,const void* b,uint32_t n){OCG_DuelSetResponse(h->duel,b,n);}
uint32_t ygo_count(Handle* h,uint8_t p,uint32_t loc){return OCG_DuelQueryCount(h->duel,p,loc);}
const void* ygo_query(Handle* h,uint8_t p,uint32_t loc,uint32_t seq,uint32_t* len) {
    OCG_QueryInfo q{};q.con=p;q.loc=loc;q.seq=seq;
    q.flags=QUERY_CODE|QUERY_POSITION|QUERY_TYPE|QUERY_LEVEL|QUERY_ATTACK|QUERY_DEFENSE|QUERY_EQUIP_CARD|QUERY_TARGET_CARD|QUERY_COUNTERS|QUERY_STATUS|QUERY_IS_PUBLIC;
    return OCG_DuelQuery(h->duel,len,&q);
}
}
