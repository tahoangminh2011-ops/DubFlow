(() => {
'use strict';
/* =========================================================
   1) CẤU HÌNH, NGÔN NGỮ, HẰNG SỐ (đồng bộ với editor.html)
   ========================================================= */
const CFG = Object.assign({ apiBase:'', token:'', plan:'MAX', pollMs:1500 }, window.LINGVO_CONFIG || {});
// Thương hiệu có thể tùy chỉnh riêng trong branding.js (window.LINGVO_BRANDING).
const BRAND = Object.assign({ name:'Lingvo', logoLight:'https://translatevideo.flowai.website/logo-mark-light.png', logoDark:'https://translatevideo.flowai.website/logo-mark-dark.png', logoAlt:'Lingvo' }, window.LINGVO_BRANDING || {});
const brandLogoLight = BRAND.logoLight || BRAND.logoDark || '';
const brandLogoDark = BRAND.logoDark || BRAND.logoLight || '';
const qs = new URLSearchParams(location.search);
[['api','apiBase'],['token','token'],['plan','plan']].forEach(([q,k]) => { const v = qs.get(q); if (v) CFG[k] = v; });
const store = { get:k=>{try{return localStorage.getItem(k)}catch(e){return null}}, set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}} };
if (CFG.token) store.set('lingvo_token', CFG.token);
const getToken = () => store.get('lingvo_token') || '';
let lang = store.get('atv_lang') || 'vi';
const t = (vi, en) => lang === 'vi' ? vi : en;
const PLAN = String(CFG.plan||'FREE').toUpperCase(); const isMax=()=>PLAN==='MAX'; const isPaid=()=>PLAN!=='FREE';
const LANGS = { auto:['Tự động','Auto','🌐'], vi:['Tiếng Việt','Vietnamese','🇻🇳'], en:['Tiếng Anh','English','🇬🇧'], ja:['Tiếng Nhật','Japanese','🇯🇵'], ko:['Tiếng Hàn','Korean','🇰🇷'], zh:['Tiếng Trung','Chinese','🇨🇳'], es:['Tiếng Tây Ban Nha','Spanish','🇪🇸'], fr:['Tiếng Pháp','French','🇫🇷'], de:['Tiếng Đức','German','🇩🇪'], pt:['Tiếng Bồ Đào Nha','Portuguese','🇵🇹'], ru:['Tiếng Nga','Russian','🇷🇺'], th:['Tiếng Thái','Thai','🇹🇭'] };
const L = c => LANGS[c] || LANGS.auto;
const PLATFORMS = [['youtube','YouTube','#ff3b3b'],['douyin','Douyin','#111'],['tiktok','TikTok','#1f2937'],['instagram','Instagram','#d6249f'],['x','X (Twitter)','#111'],['bilibili','Bilibili','#00a1d6'],['kuaishou','Kuaishou','#ff4906'],['xiaohongshu','Xiaohongshu','#ff2442'],['weibo','Weibo','#e6162d'],['reddit','Reddit','#ff4500'],['toutiao','Toutiao','#e0342b'],['pearvideo','PearVideo','#f5a623']];
const STT_PROVIDERS = [ { id:'groq', name:'Groq Whisper', vi:'Whisper large-v3, độ chính xác cao, đa ngôn ngữ', en:'Whisper large-v3, high accuracy, multilingual', tags:['free'], key:'groq' }, { id:'gemini', name:'Gemini STT', vi:'Nhận dạng giọng nói bằng mô hình Gemini của Google', en:'Speech recognition powered by Google Gemini', tags:['free','trial'], key:'gemini' } ];
const TX_PROVIDERS = [ { id:'groq', name:'Groq LLM', vi:'Dịch nhanh, giữ ngữ cảnh tốt', en:'Fast translation with strong context', tags:['free'], key:'groq' }, { id:'xkira', name:'xKiro Qwen', vi:'Dùng catalog Qwen live từ xKiro', en:'Use the live Qwen catalog from xKiro', tags:['free'], key:'xkira' }, { id:'gemini', name:'Gemini LLM', vi:'Đa ngôn ngữ, ngữ cảnh dài', en:'Multilingual with long context', tags:['free','trial'], key:'gemini' } ];
const OCR_PROVIDERS = [ { id:'local', name:'PP-OCRv5 Mobile · VideoOCR V2', vi:'OCR local ưu tiên recall, adaptive sampling, GPU/CPU fallback và timeline theo frame.', en:'Recall-first local OCR with adaptive sampling, GPU/CPU fallback and frame-accurate timeline.', tags:['local','high-recall'], key:'local' } ];
const TTS_PROVIDERS = [ { id:'xkira_tts', name:'xKiro TTS', vi:'Giọng AI xKiro, danh sách giọng thật từ API', en:'xKiro AI voices, live catalog from the API', tags:['free'], key:'xkira' }, { id:'edge_tts', name:'Microsoft Edge TTS', vi:'Miễn phí, giọng tự nhiên, nhiều ngôn ngữ', en:'Free, natural voices in many languages', tags:['free'] } ];
const EDGE_VOICES = { vi:[['vi-VN-HoaiMyNeural','Hoài My','F'],['vi-VN-NamMinhNeural','Nam Minh','M']], en:[['en-US-AvaNeural','Ava','F'],['en-US-AndrewNeural','Andrew','M']], ja:[['ja-JP-NanamiNeural','Nanami','F']], ko:[['ko-KR-SunHiNeural','Sun-Hi','F']], zh:[['zh-CN-XiaoxiaoNeural','Xiaoxiao','F']], es:[['es-ES-ElviraNeural','Elvira','F']], fr:[['fr-FR-DeniseNeural','Denise','F']], de:[['de-DE-KatjaNeural','Katja','F']], pt:[['pt-PT-RaquelNeural','Raquel','F']], ru:[['ru-RU-SvetlanaNeural','Svetlana','F']], th:[['th-TH-PremwadeeNeural','Premwadee','F']] };
let XKIRA_VOICES = [];
const voicesFor = c => EDGE_VOICES[c] || EDGE_VOICES.en;
const xkiraVoiceOptions = () => XKIRA_VOICES.map(v=>[v.id, `${v.name}${v.gender?' · '+(v.gender==='female'?'Nữ':'Nam'):''}`]);
const xkiraVoiceCards = (selected,onSelect,onPreview) => h('div',{style:{display:'grid',gridTemplateColumns:'repeat(4,minmax(0,1fr))',gap:'8px'}}, XKIRA_VOICES.slice(0,4).map(v=>h('button',{type:'button',class:'opt-card','data-on':selected===v.id,onClick:()=>onSelect(v.id),style:{minHeight:'90px',textAlign:'left',padding:'10px'}},h('div',{style:{display:'flex',justifyContent:'space-between',alignItems:'center',gap:'6px'}},h('b',{class:'trunc'},v.name||v.id),btn({icon:'volume',title:t('Nghe thử','Preview'),onClick:e=>{e.stopPropagation();onPreview(v.id);}})),h('small',{class:'muted'},v.gender==='female'?t('Nữ','Female'):v.gender==='male'?t('Nam','Male'):''),v.locale?h('small',{class:'muted'},v.locale):null)));
function selectedVoiceForProject(pv){ return pv.xkiraVoice || pv.voice || pv.edgeVoice || ''; }
const STATUS = { QUEUED:{vi:'Trong hàng đợi',en:'Queued',b:'neutral'}, PROCESSING:{vi:'Đang xử lý',en:'Processing',b:'cyan'}, TRANSCRIBED:{vi:'Sẵn sàng chỉnh sửa',en:'Ready to edit',b:'cyan'}, TRANSLATED:{vi:'Đã dịch',en:'Translated',b:'cyan'}, DUBBED:{vi:'Đã lồng tiếng',en:'Dubbed',b:'cyan'}, COMPLETED:{vi:'Hoàn thành',en:'Completed',b:'success'}, FAILED:{vi:'Thất bại',en:'Failed',b:'destructive'}, CANCELLED:{vi:'Đã hủy',en:'Cancelled',b:'neutral'} };

/* =========================================================
   2) API — HTTP thật (dưới apiBase) hoặc demo
   GET  /projects            GET /projects/:id           DELETE /projects/:id
   GET  /videos               POST /videos/import-url {url}      POST /videos/upload (multipart)
   POST /projects {video_id,source_lang,target_lang,subtitle_source,ocr_region,want_dubbing,filter_transcript_noise,providers}
   POST /projects/:id/transcribe
   GET  /settings              PATCH /settings/providers          PUT /settings/api-keys/:provider {api_key}
   GET  /billing/quota          GET /presence/online-count
   ========================================================= */
const num=(v,d=0)=>typeof v==='number'?v:d, str=(v,d='')=>typeof v==='string'?v:d;
const normalizeStatus=s=>({"Đang chờ xử lý":'QUEUED',"Đang xử lý":'PROCESSING',"Đang tách âm thanh":'PROCESSING',"Đang nhận diện giọng nói":'PROCESSING',"Đang dịch":'PROCESSING',"Đang tạo giọng đọc":'PROCESSING',"Đang render":'PROCESSING',"Sẵn sàng chỉnh sửa":'TRANSCRIBED',"Đã dịch":'TRANSLATED',"Đã lồng tiếng":'DUBBED',"Hoàn tất":'COMPLETED',"Lỗi":'FAILED',"Đã hủy":'CANCELLED'}[s]||s||'QUEUED');
const mapProject = r => ({ id:str(r.project_id||r.id), name:str(r.video_name||r.name), thumbnail:r.thumbnail_url||r.thumbnail||'', src:str(r.source_lang,'en'), tgt:str(r.target_lang,'vi'), status:normalizeStatus(r.status), stage:r.stage||null, progress:num(r.progress), queuePosition:r.queue_position??null, durationMin:num(r.video_duration_minutes), createdAt:str(r.created_at,new Date().toISOString()), error:r.error_message||r.error||null, translationMissingSegments:Array.isArray(r.translation_missing_segments)?r.translation_missing_segments:[] });
function httpApi(base){
  base = base.replace(/\/$/,'');
  const req = async (path, init={}) => {
    const hd=new Headers(init.headers||{});
    if(init.body && typeof init.body==='string') hd.set('Content-Type','application/json');
    const tk=getToken(); if(tk) hd.set('Authorization','Bearer '+tk);
    const r=await fetch(base+'/api'+path,Object.assign({},init,{headers:hd}));
    if(!r.ok){ let m='HTTP '+r.status; try{const j=await r.json(); m=j.message||j.detail||m;}catch(e){} throw new Error(m); }
    return r.status===204?null:r.json();
  };
  const J=o=>JSON.stringify(o);
  const langName=(code)=>{
    const m={auto:'Tự động nhận diện',vi:'Tiếng Việt',en:'Tiếng Anh',zh:'Tiếng Trung',ja:'Tiếng Nhật',ko:'Tiếng Hàn'};
    return m[code]||code||'Tự động nhận diện';
  };
  const formProject=(o)=>{
    const fd=new FormData();
    fd.append('name',o.video_name||o.name||'Lingvo project');
    fd.append('src_lang',langName(o.source_lang));
    fd.append('dst_lang',langName(o.target_lang));
    fd.append('voice',o.voice||o.providers?.voice||'female_sweet');
    fd.append('want_subtitle',String(o.want_subtitle!==false));
    fd.append('want_dubbing',String(o.want_dubbing!==false));
    fd.append('subtitle_source',o.subtitle_source||'stt');
    fd.append('filter_transcript_noise',String(!!o.filter_transcript_noise));
    fd.append('ocr_region',JSON.stringify(o.ocr_region||{x:0,y:65,width:100,height:35}));
    fd.append('keep_original_audio',String(!!o.keep_original_audio));
    fd.append('stt_provider',o.stt_provider||o.providers?.stt||'groq');
    fd.append('translate_provider',o.translate_provider||o.providers?.translate||'groq');
    fd.append('tts_provider',o.tts_provider||o.providers?.tts||'xkira_tts');
    fd.append('stt_model',o.stt_model||o.providers?.sttModel||'');
    fd.append('ocr_model',o.ocr_model||o.providers?.ocrModel||'');
    fd.append('translate_model',o.translate_model||o.providers?.translateModel||'');
    fd.append('tts_model',o.tts_model||o.providers?.ttsModel||'xkiro-voice');
    fd.append('full_model',o.full_model||o.providers?.fullModel||'');
    fd.append('ocr_provider',o.ocr_provider||o.providers?.ocr||'local');
    fd.append('translate_refine',String(!!o.translate_refine));
    if(o.import_id) fd.append('import_id',o.import_id);
    if(o.url) fd.append('url',o.url);
    if(o.file) fd.append('file',o.file,o.file.name);
    return fd;
  };
  const reqForm=async(path,fd)=>{
    const tk=getToken(); const hd={}; if(tk) hd.Authorization='Bearer '+tk;
    const r=await fetch(base+'/api'+path,{method:'POST',headers:hd,body:fd});
    if(!r.ok){let m='HTTP '+r.status;try{const j=await r.json();m=j.message||j.detail||m;}catch(e){}throw new Error(m);}
    return r.json();
  };
  return {
    listProjects:()=>req('/projects').then(d=>(Array.isArray(d)?d:d.projects||[]).map(mapProject)),
    getProject:id=>req(`/projects/${id}`).then(mapProject),
    deleteProject:id=>req(`/projects/${id}`,{method:'DELETE'}),
    importUrl:async url=>req('/videos/import-url',{method:'POST',body:J({url})}).then(v=>Object.assign(v,{video_id:v.id,_url:v.webpage_url||url,_import_id:v.id,name:v.title,duration_minutes:v.duration_minutes,thumbnail:v.thumbnail})),
    uploadVideo:async(file,onProgress)=>{ if(onProgress) onProgress(100); return {video_id:'upload_'+Date.now(),name:file.name,thumbnail:'',duration_minutes:0,platform:'upload',_file:file}; },
    createProject:o=>reqForm('/projects',formProject(o)).then(mapProject),
    startTranscribe:id=>req(`/projects/${id}/process`,{method:'POST',body:'{}'}),
    getSettings:async()=>{
      const [keys,settings]=await Promise.all([
        req('/settings/keys').catch(()=>({})),
        req('/settings/providers').catch(()=>({stt_provider:'groq',ocr_provider:'local',translate_provider:'groq',tts_provider:'xkira_tts',translate_refine:false,xkiraVoice:'',stt_model:'',ocr_model:'',translate_model:'',tts_model:'xkiro-voice',full_model:''}))
      ]);
      return {providers:Object.assign({stt:'groq',ocr:'local',translate:'groq',tts:'xkira_tts',refine:false,xkiraVoice:'',sttModel:'',ocrModel:'PP-OCRv5-mobile',translateModel:'',ttsModel:'xkiro-voice',fullModel:'',ocr_mode:'HIGH_RECALL'},settings||{}),keys};
    },
    saveProviders:o=>req('/settings/providers',{method:'POST',body:J(o)}),
    saveApiKey:(provider,key)=>req('/settings/keys',{method:'POST',body:J({[provider]:key})}),
    getQuota:()=>req('/quota'),
    getOnlineCount:()=>Promise.resolve(1),
    getXkiraVoices:async(refresh=false)=>req(`/tts/voices?language=vi&limit=4${refresh?'&refresh=1':''}`).catch(()=>[]),
    getModels:async(modality='all')=>req(`/models?modality=${encodeURIComponent(modality)}`).then(r=>Array.isArray(r)?r:(r.data||[])),
    previewVoice:async(voice)=>{ const r=await fetch(base+'/api/tts/preview',{method:'POST',headers:{'Content-Type':'application/json',...(getToken()?{Authorization:'Bearer '+getToken()}: {})},body:J({text:'Xin chào, đây là giọng đọc thử của Lingvo.',voice,provider:'xkira_tts',language:'vi',rate:1.3,pitch:0})}); if(!r.ok) throw new Error('Không tạo được audio thử'); return URL.createObjectURL(await r.blob()); },
  };
}
function demoApi(){
  const store2 = { projects:[], settings:{ providers:{stt:'groq',translate:'groq',tts:'xkira_tts',refine:false,edgeVoice:'vi-VN-HoaiMyNeural',xkiraVoice:'',speed:1}, keys:{openai:false,gemini:true,groq:false,elevenlabs:false} }, quota:{used:0,total:30} };
  const wait=ms=>new Promise(r=>setTimeout(r,ms)); const clone=v=>JSON.parse(JSON.stringify(v));
  const THUMB='https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400&q=60';
  function tick(p){ if(!p._job) return; p.progress=Math.min(100,p.progress+14); if(p.progress>30) p.stage='STT'; if(p.progress>=100){ p.status='TRANSCRIBED'; p.stage=null; p._job=false; store2.quota.used=Math.min(store2.quota.total,store2.quota.used+p.durationMin); return; } setTimeout(()=>tick(p),650); }
  return {
    async listProjects(){ await wait(150); return clone(store2.projects); },
    async getProject(id){ await wait(80); const p=store2.projects.find(x=>x.id===id); if(!p) throw new Error(t('Không tìm thấy dự án','Project not found')); return clone(p); },
    async deleteProject(id){ await wait(100); store2.projects=store2.projects.filter(p=>p.id!==id); },
    async importUrl(url){ await wait(900); const platform=PLATFORMS.find(p=>url.toLowerCase().includes(p[0]))||PLATFORMS[0]; return { video_id:'vid_'+Date.now(), name:(platform[1])+' — '+t('video mẫu đã tải về','sample imported video'), thumbnail:THUMB, duration_minutes:3.4, platform:platform[0] }; },
    async uploadVideo(file,onProgress){ for(let p=0;p<=100;p+=20){ await wait(150); onProgress&&onProgress(p);} return { video_id:'vid_'+Date.now(), name:file.name, thumbnail:'', duration_minutes:Math.max(0.5,Math.round(file.size/1e7*10)/10) }; },
    async createProject(o){ await wait(200); const p=mapProject({ project_id:'prj_'+Date.now(), video_name:o.video_name||t('Dự án mới','New project'), source_lang:o.source_lang, target_lang:o.target_lang, status:'QUEUED', video_duration_minutes:o.duration_minutes||2, created_at:new Date().toISOString() }); p.thumbnail=o.thumbnail||''; p._job=false; store2.projects.unshift(p); return clone(p); },
    async startTranscribe(id){ await wait(150); const p=store2.projects.find(x=>x.id===id); if(!p) return; p.status='PROCESSING'; p.stage='EXTRACT_AUDIO'; p.progress=0; p._job=true; setTimeout(()=>tick(p),500); },
    async getSettings(){ await wait(80); return clone(store2.settings); },
    async saveProviders(o){ await wait(100); Object.assign(store2.settings.providers,o); },
    async saveApiKey(provider,key){ await wait(150); store2.settings.keys[provider]=!!key; },
    async getQuota(){ await wait(80); return clone(store2.quota); },
    async getOnlineCount(){ return 1; },
    _store:store2,
  };
}
const demoApiInstance = demoApi();
const httpApiInstance = CFG.apiBase ? httpApi(CFG.apiBase) : null;
let api = null;

/* =========================================================
   3) STATE & DOM HELPERS (giống editor.html)
   ========================================================= */
const S = { route:'dashboard', conn:'offline', connError:'', connBarClosed:false, projects:[], settings:null, quota:null, online:1, bannerClosed:store.get('atv_banner_closed')==='1', mobileNav:false,
  wiz:{ step:0, source:'link', url:'', urlBusy:false, video:null, uploadProgress:0, srcLang:'auto', tgtLang:'vi', subtitleSource:'stt', wantDubbing:true, filterNoise:false, ocrRegion:{x:0,y:65,w:100,h:35}, providers:{stt:'groq',ocr:'local',translate:'groq',tts:'xkira_tts',refine:false,voice:'',xkiraVoice:'',sttModel:'',ocrModel:'PP-OCRv5-mobile',translateModel:'',ttsModel:'xkiro-voice',fullModel:'',ocr_mode:'HIGH_RECALL'}, creating:false },
  processingId:null };
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>Array.from(r.querySelectorAll(s)), clamp=(v,a,b)=>Math.min(Math.max(v,a),b);
const isMobile=()=>window.innerWidth<=900;
function h(tag,attrs,...kids){ const el=document.createElement(tag); if(attrs) for(const [k,v] of Object.entries(attrs)){ if(v==null||v===false) continue; if(k==='class') el.className=v; else if(k==='style'&&typeof v==='object') Object.assign(el.style,v); else if(k.startsWith('on')&&typeof v==='function') el.addEventListener(k.slice(2).toLowerCase(),v); else if(v===true) el.setAttribute(k,''); else el.setAttribute(k,v);} for(const c of kids.flat(Infinity)){ if(c==null||c===false) continue; el.append(c instanceof Node?c:document.createTextNode(String(c))); } return el; }
const IC={dashboard:'<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',plus:'<path d="M5 12h14"/><path d="M12 5v14"/>',video:'<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/>',folder:'<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',box:'<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="M3.3 7 12 12l8.7-5"/><path d="M12 22V12"/>',send:'<path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z"/><path d="m21.854 2.147-10.94 10.939"/>',settings:'<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',gauge:'<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',creditCard:'<rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/>',lifeBuoy:'<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m4.93 19.07 4.24-4.24"/>',bookOpen:'<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>',arrowLeft:'<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>',arrowRight:'<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',link:'<path d="M9 17H7A5 5 0 0 1 7 7h2"/><path d="M15 7h2a5 5 0 1 1 0 10h-2"/><line x1="8" x2="16" y1="12" y2="12"/>',upload:'<path d="M12 3v12"/><path d="m17 8-5-5-5 5"/><path d="M5 21h14"/>',film:'<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>',globe:'<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',x:'<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',check:'<path d="M20 6 9 17l-5-5"/>',info:'<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',ok:'<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',err:'<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',alert:'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',clock:'<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',bell:'<path d="M10.268 21a2 2 0 0 0 3.464 0"/><path d="M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326"/>',chevDown:'<path d="m6 9 6 6 6-6"/>',rocket:'<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>',search:'<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',trash:'<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',mic:'<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',languages:'<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>',volume:'<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>',captions:'<rect width="18" height="14" x="3" y="5" rx="2"/><path d="M7 15h4M15 15h2M7 11h2M13 11h4"/>',loader:'<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',moon:'<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',menu:'<line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/>',refresh:'<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',sparkles:'<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',mail:'<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>',play:'<path d="M6 3 20 12 6 21z"/>'};
function ic(n,s=16,c=''){ const e=document.createElementNS('http://www.w3.org/2000/svg','svg'); e.setAttribute('viewBox','0 0 24 24'); e.setAttribute('width',s); e.setAttribute('height',s); e.setAttribute('fill','none'); e.setAttribute('stroke','currentColor'); e.setAttribute('stroke-width','2'); e.setAttribute('stroke-linecap','round'); e.setAttribute('stroke-linejoin','round'); e.innerHTML=IC[n]||''; if(c) e.setAttribute('class',c); return e; }
function btn({v='ghost',sm=true,icon,label,title,onClick,disabled,cls='',busy,attrs}){ const b=h('button',Object.assign({type:'button',class:`btn btn-${v} ${sm?'btn-sm':''} ${label?'':'btn-icon'} ${cls}`,title,disabled:!!disabled,onClick},attrs||{})); if(busy) b.append(ic('loader',sm?14:16,'spin')); else if(icon) b.append(ic(icon,sm?14:16)); if(label) b.append(h('span',null,label)); return b; }
const sw=(on,fn)=>h('button',{type:'button',class:'switch',role:'switch','data-on':on,onClick:e=>{e.stopPropagation();fn(!e.currentTarget.hasAttribute('data-on'));}},h('i'));
const sel=(opts,val,fn,attrs)=>{ const s=h('select',Object.assign({class:'input',onChange:e=>fn(e.target.value)},attrs||{})); opts.forEach(([v,l])=>s.append(h('option',{value:v},l))); s.value=val; return s; };
const fmt=(min)=>{ const m=Math.floor(min); const s=Math.round((min-m)*60); return `${m}:${String(s).padStart(2,'0')}`; };
function toast(msg,variant='info'){ const col={success:'var(--success)',error:'var(--destructive)',warning:'var(--warning)',info:'var(--primary)'}[variant]; const icn=ic({success:'ok',error:'err',warning:'alert',info:'info'}[variant],17); icn.style.color=col; icn.style.flexShrink='0'; const el=h('div',{class:'toast',style:{borderLeftColor:col}},icn,h('div',{class:'grow'},msg),btn({icon:'x',onClick:()=>el.remove()})); $('#toasts').append(el); setTimeout(()=>{el.classList.add('out');setTimeout(()=>el.remove(),220);},4200); }
function dialog({title,desc,body,footer,onClose,width}){ const root=$('#layers'); const wrap=h('div',{class:'overlay',onMousedown:e=>{ if(e.target===e.currentTarget) close(); }}); const close=()=>{ wrap.remove(); document.removeEventListener('keydown',esc); onClose&&onClose(); }; const esc=e=>{ if(e.key==='Escape') close(); }; document.addEventListener('keydown',esc); const dlg=h('div',{class:'dialog',role:'dialog',style:width?{maxWidth:width+'px'}:null},h('div',{style:{display:'flex',justifyContent:'space-between',alignItems:'flex-start',gap:'10px'}},h('div',null,h('div',{class:'dlg-title'},title),desc?h('div',{class:'dlg-desc'},desc):null),btn({icon:'x',onClick:close}))); if(body) dlg.append(h('div',{style:{marginTop:'16px'}},body)); if(footer) dlg.append(h('div',{class:'dlg-foot'},...footer(close))); wrap.append(dlg); root.append(wrap); return close; }
const confirmDlg=({title,desc,confirmLabel,variant='destructive'})=>new Promise(res=>{ dialog({title,desc,footer:close=>[btn({v:'ghost',sm:false,label:t('Hủy','Cancel'),onClick:()=>{close();res(false);}}),btn({v:variant,sm:false,label:confirmLabel,onClick:()=>{close();res(true);}})],onClose:()=>res(false)}); });
function emptyState(icon,title,desc,showDemoBtn){ return h('div',{class:'empty'}, h('div',{class:'eic'},ic(icon,20)), h('b',null,title), desc?h('p',null,desc):null, showDemoBtn?btn({v:'outline',icon:'sparkles',label:t('Xem dữ liệu minh hoạ','Preview sample data'),onClick:()=>loadDemoPreview()}):null); }
function providerCard(pr,cur,onSel,keys){ const needsKey=!pr.tags.includes('free'); const saved=pr.key?!!keys[pr.key]:false; const enabled=!needsKey||saved; const tags=[]; if(pr.tags.includes('free')) tags.push(h('span',{class:'tag tag-free'},t('Miễn phí','Free'))); if(pr.tags.includes('trial')) tags.push(h('span',{class:'tag tag-trial'},'Trial')); if(pr.key) tags.push(h('span',{class:'tag tag-key'},saved?t('Có key','Key saved'):'API key')); return h('button',{type:'button',class:'provider-card','data-on':cur===pr.id,disabled:!enabled,title:enabled?'':t('Thêm API key trong Cài đặt để dùng nhà cung cấp này','Add an API key in Settings to use this provider'),onClick:()=>onSel(pr.id)},h('span',{class:'rd'}),h('span',{class:'grow'},h('b',null,pr.name,...tags),h('small',null,lang==='vi'?pr.vi:pr.en))); }

/* =========================================================
   4) SIDEBAR / TOPBAR / KẾT NỐI
   ========================================================= */
const NAV=[ {group:t('KHÔNG GIAN LÀM VIỆC','WORKSPACE'),items:[ ['dashboard','dashboard',t('Bảng điều khiển','Dashboard')], ['new','plus',t('Dự án mới','New project')], ['videos','video',t('Video','Videos')], ['projects','folder',t('Dự án','Projects')], ['outputs','box',t('Sản phẩm','Outputs')], ['autopublish','send',t('Tự động đăng bài','Auto-publish'),t('sớm','soon')] ]},
  {group:t('TÀI KHOẢN','ACCOUNT'),items:[ ['settings','settings',t('Cài đặt','Settings')], ['quota','gauge',t('Hạn ngạch','Quota')], ['billing','creditCard',t('Gói & Thanh toán','Plans & Billing')], ['support','lifeBuoy',t('Hỗ trợ','Support')], ['tutorial','bookOpen',t('Hướng dẫn','Tutorial')] ]} ];
const PAGE_TITLE={ dashboard:t('Bảng điều khiển','Dashboard'), new:t('Dự án mới','New project'), processing:t('Đang xử lý','Processing'), videos:t('Video','Videos'), projects:t('Dự án','Projects'), outputs:t('Sản phẩm','Outputs'), autopublish:t('Tự động đăng bài','Auto-publish'), settings:t('Cài đặt','Settings'), quota:t('Hạn ngạch','Quota'), billing:t('Gói & Thanh toán','Plans & Billing'), support:t('Hỗ trợ','Support'), tutorial:t('Hướng dẫn','Tutorial') };
function go(route,opts){ S.route=route; if(opts&&opts.projectId) S.processingId=opts.projectId; if(route==='new'&&opts&&opts.tab) S.wiz.source=opts.tab; S.mobileNav=false; renderAll(); window.scrollTo(0,0); $('#content')&&($('#content').scrollTop=0); }
function renderSidebar(){ const sb=$('#sidebar'); sb.innerHTML=''; sb.toggleAttribute('data-open',S.mobileNav);
  sb.append(h('div',{class:'sb-logo'},brandLogoLight?h('img',{class:'lm-light',src:brandLogoLight,alt:BRAND.logoAlt||BRAND.name}):null,brandLogoDark?h('img',{class:'lm-dark',src:brandLogoDark,alt:BRAND.logoAlt||BRAND.name}):null,h('b',null,BRAND.name||'Lingvo')));
  NAV.forEach(g=>{ const wrap=h('div',{class:'sb-group'},h('div',{class:'sb-title'},g.group)); g.items.forEach(([route,icon,label,tag])=>{ const item=h('button',{type:'button',class:'sb-item','data-on':S.route===route,onClick:()=>go(route)},ic(icon,16),h('span',{class:'grow trunc'},label)); if(tag) item.append(h('span',{class:'tag tag-trial'},tag)); wrap.append(item); }); sb.append(wrap); });
  sb.append(h('div',{class:'sb-foot'},btn({icon:'arrowLeft',label:t('Về trang chủ','Back to home'),cls:'btn-block',v:'ghost',attrs:{style:'justify-content:flex-start'},onClick:()=>notifyParent('lingvo:open-landing')})));
}
function renderTopbar(){ const top=$('#topbar'); top.innerHTML=''; const m=isMobile();
  if(m) top.append(btn({icon:'menu',onClick:()=>{S.mobileNav=true;renderSidebar();$('#sbScrim').hidden=false;}}));
  top.append(h('h1',null,PAGE_TITLE[S.route]||''));
  const act=h('div',{class:'topbar-actions'});
  if(S.route==='dashboard'){ act.append(btn({v:'outline',icon:'link',label:m?null:t('Nhập link','Import link'),onClick:()=>go('new',{tab:'link'})}), btn({v:'primary',icon:'plus',label:m?null:t('Dự án mới','New project'),onClick:()=>go('new')})); }
  act.append(h('span',{class:'online-pill'},h('i'),`${S.online} ${t('đang online','online')}`));
  if(S.connBarClosed){ const dotState={offline:'off',checking:'checking',connected:'on',error:'off',demo:'demo'}[S.conn]||'off'; act.append(h('button',{type:'button',class:'conn-dot-btn',title:t('Xem trạng thái kết nối backend','View backend connection status'),onClick:()=>{S.connBarClosed=false;renderConnBar();}},h('span',{class:'conn-dot','data-s':dotState}))); }
  act.append(btn({icon:'globe',label:lang.toUpperCase(),onClick:()=>{lang=lang==='vi'?'en':'vi';store.set('atv_lang',lang);location.reload();}}));
  act.append(btn({icon:document.documentElement.classList.contains('dark')?'sun':'moon',onClick:()=>{const d=document.documentElement.classList.toggle('dark');store.set('atv_dark',String(d));renderTopbar();}}));
  act.append(h('div',{class:'avatar'},'AD'));
  top.append(act);
}
function renderConnBar(){ const bar=$('#connBar'); if(!bar) return; bar.innerHTML=''; const cfgLabel=CFG.apiBase||t('(chưa cấu hình)','(not configured)');
  const map={ offline:{s:'off',label:t('Chưa kết nối backend','Backend not connected'),sub:t('Đặt LINGVO_CONFIG.apiBase để tải dữ liệu thật.','Set LINGVO_CONFIG.apiBase to load real data.')},
    checking:{s:'checking',label:t('Đang kết nối…','Connecting…'),sub:cfgLabel},
    connected:{s:'on',label:t('Đã kết nối backend','Backend connected'),sub:cfgLabel},
    error:{s:'off',label:t('Không thể kết nối backend','Could not reach the backend'),sub:S.connError||cfgLabel},
    demo:{s:'demo',label:t('Đang xem dữ liệu minh hoạ','Previewing sample data'),sub:t('Chưa kết nối backend thật.','No real backend connected.')} };
  const info=map[S.conn]||map.offline; bar.hidden=!!S.connBarClosed; if(S.connBarClosed) return;
  bar.append(h('span',{class:'conn-dot','data-s':info.s}),h('span',{class:'conn-label'},info.label),h('span',{class:'conn-sub mono trunc'},info.sub),h('div',{class:'grow'}));
  if(S.conn==='offline') bar.append(btn({v:'outline',icon:'sparkles',label:t('Xem dữ liệu minh hoạ','Preview sample data'),onClick:()=>loadDemoPreview()}));
  if(S.conn==='error') bar.append(btn({v:'outline',icon:'refresh',label:t('Thử lại','Retry'),onClick:()=>connectBackend()}),btn({v:'ghost',icon:'sparkles',label:t('Xem dữ liệu minh hoạ','Preview sample data'),onClick:()=>loadDemoPreview()}));
  if(S.conn==='demo'&&CFG.apiBase) bar.append(btn({v:'outline',icon:'refresh',label:t('Kết nối backend thật','Connect real backend'),onClick:()=>connectBackend()}));
  bar.append(h('button',{type:'button',class:'conn-close',title:t('Đóng thông báo','Dismiss notice'),onClick:()=>{S.connBarClosed=true;renderConnBar();renderTopbar();}},ic('x',14)));
}
function notifyParent(type,payload){
  try{
    if(type==='lingvo:open-editor'){
      if(payload&&payload.projectId) store.set('lingvo_project_id', payload.projectId);
      const p=new URLSearchParams(location.search);
      p.set('page','editor'); p.set('project',(payload&&payload.projectId)||'');
      location.href=location.pathname+'?'+p.toString();
      return;
    }
    if(type==='lingvo:open-app'){
      const p=new URLSearchParams(location.search);
      p.set('page','app'); p.delete('project');
      location.href=location.pathname+'?'+p.toString();
      return;
    }
  }catch(e){}
}
async function connectBackend(){ if(!CFG.apiBase){ S.conn='offline'; renderConnBar(); return; } S.conn='checking'; S.connError=''; renderConnBar();
  try{ const ctrl=new AbortController(); const timer=setTimeout(()=>ctrl.abort(),6000); try{ await fetch(CFG.apiBase+'/api/health',{method:'GET',signal:ctrl.signal}); } catch(err){ throw new Error(err&&err.name==='AbortError'?t('Hết thời gian chờ kết nối','Connection timed out'):t('Không thể kết nối tới máy chủ backend','Could not reach the backend server')); } finally{ clearTimeout(timer); }
    S.conn='connected'; renderConnBar(); api=httpApiInstance; await loadWorkspace(); }
  catch(e){ S.conn='error'; S.connError=e.message; renderConnBar(); renderAll(); } }
function loadDemoPreview(){ S.conn='demo'; renderConnBar(); api=demoApiInstance; loadWorkspace(); }
async function loadWorkspace(){ try{ const [projects,settings,online,voices] = await Promise.all([api.listProjects(),api.getSettings(),api.getOnlineCount(),api.getXkiraVoices().catch(()=>[])]); S.projects=projects; S.settings=settings; S.online=online||1; XKIRA_VOICES=Array.isArray(voices)?voices.slice(0,4):[]; }catch(e){ toast(e.message||t('Không tải được dữ liệu workspace','Could not load workspace data'),'error'); } renderAll(); }

/* =========================================================
   5) TRANG DASHBOARD
   ========================================================= */
function pageDashboard(root){
  // Dashboard gọn: bỏ banner lưu trữ, bảng cập nhật, roadmap và hạn mức hiển thị.
  // Logo và tên thương hiệu được cấu hình riêng trong branding.js.
  root.append(h('div',{class:'card',style:{display:'flex',alignItems:'center',gap:'14px',padding:'22px 24px',marginBottom:'18px'}},
    brandLogoLight?h('img',{class:'lm-light',src:brandLogoLight,alt:BRAND.logoAlt||BRAND.name,style:{width:'48px',height:'48px',objectFit:'contain'}}):null,
    brandLogoDark?h('img',{class:'lm-dark',src:brandLogoDark,alt:BRAND.logoAlt||BRAND.name,style:{width:'48px',height:'48px',objectFit:'contain'}}):null,
    h('div',{class:'grow'},h('h2',{style:{margin:'0',fontSize:'22px',fontWeight:'750'}},BRAND.name||'Lingvo'))
  ));
  root.append(h('div',{class:'section-head'},h('h2',null,t('Dự án gần đây','Recent projects')),S.projects.length?h('a',{href:'javascript:void(0)',onClick:()=>go('projects')},t('Xem tất cả','View all'),ic('arrowRight',13)):null));
  if(!S.projects.length) root.append(h('div',{class:'card'},emptyState('folder',t('Chưa có dự án nào. Hãy tạo dự án đầu tiên!','No projects yet. Create your first one!'),null,false),h('div',{style:{textAlign:'center',paddingBottom:'26px'}},btn({v:'primary',sm:false,icon:'plus',label:t('Dự án mới','New project'),onClick:()=>go('new')}))));
  else { const grid=h('div',{class:'proj-grid'}); S.projects.slice(0,8).forEach(p=>grid.append(projectCard(p))); root.append(grid); }
}
function statusBadge(p){ const st=STATUS[p.status]||STATUS.QUEUED; return h('span',{class:'badge badge-'+st.b},lang==='vi'?st.vi:st.en); }
function projectCard(p){ const busy=p.status==='PROCESSING'||p.status==='QUEUED'; return h('div',{class:'card proj-card',onClick:()=>openProject(p)},
  h('div',{class:'proj-thumb'}, p.thumbnail?h('img',{src:p.thumbnail,alt:''}):h('div',{style:{display:'grid',placeItems:'center',height:'100%',color:'var(--muted-foreground)'}},ic('film',22)), busy?h('div',{class:'pb'},h('i',{style:{width:p.progress+'%'}})):null),
  h('div',{class:'proj-body'},h('div',{class:'nm'},p.name),h('div',{class:'proj-meta'},h('span',null,`${L(p.src)[2]} → ${L(p.tgt)[2]}`),h('span',null,'·'),statusBadge(p)))); }
function openProject(p){ if((p.status==='PROCESSING'||p.status==='QUEUED')&&p.subtitleSource!=='ocr'){ go('processing',{projectId:p.id}); } else if(p.status==='FAILED'||p.status==='CANCELLED'){ toast(t('Dự án này chưa sẵn sàng để chỉnh sửa.','This project is not ready to edit.'),'warning'); } else { notifyParent('lingvo:open-editor',{projectId:p.id}); } }

/* =========================================================
   6) TRANG DANH SÁCH: VIDEO / DỰ ÁN / SẢN PHẨM
   ========================================================= */
function pageProjects(root){ root.append(h('div',{class:'section-head'},h('h2',null,t('Tất cả dự án','All projects')),btn({v:'primary',icon:'plus',label:t('Dự án mới','New project'),onClick:()=>go('new')})));
  if(!S.projects.length){ root.append(h('div',{class:'card'},emptyState('folder',t('Chưa có dự án nào','No projects yet'),t('Tạo dự án đầu tiên từ một link video hoặc file tải lên.','Create your first project from a video link or an uploaded file.'),false))); return; }
  const table=h('table',{class:'table'},h('thead',{},h('tr',null,h('th',null,t('Dự án','Project')),h('th',null,t('Ngôn ngữ','Languages')),h('th',null,t('Trạng thái','Status')),h('th',null,t('Thời lượng','Duration')),h('th',null,''))),h('tbody'));
  const tb=$('tbody',table)||table.querySelector('tbody');
  S.projects.forEach(p=>tb.append(h('tr',{class:'row-click',onClick:()=>openProject(p)},h('td',null,h('div',{style:{display:'flex',alignItems:'center',gap:'8px'}},h('div',{style:{width:'42px',height:'26px',borderRadius:'5px',background:'var(--muted)',flexShrink:0,overflow:'hidden'}},p.thumbnail?h('img',{src:p.thumbnail,style:{width:'100%',height:'100%',objectFit:'cover'}}):null),h('span',{class:'trunc'},p.name))),h('td',null,`${L(p.src)[2]} → ${L(p.tgt)[2]}`),h('td',null,statusBadge(p)),h('td',null,`${p.durationMin.toFixed(1)} ${t('phút','min')}`),h('td',{style:{textAlign:'right'}},btn({icon:'trash',title:t('Xóa dự án','Delete project'),onClick:async e=>{e.stopPropagation(); if(!await confirmDlg({title:t('Xóa dự án','Delete project'),desc:t('Dự án và toàn bộ dữ liệu liên quan sẽ bị xóa vĩnh viễn.','The project and all related data will be permanently deleted.'),confirmLabel:t('Xóa','Delete')})) return; try{ await api.deleteProject(p.id); S.projects=S.projects.filter(x=>x.id!==p.id); renderPage(); toast(t('Đã xóa dự án','Project deleted'),'success'); }catch(err){ toast(err.message,'error'); } }})))));
  root.append(h('div',{class:'card'},table)); }
function pageVideos(root){ const uploaded=S.projects.length; root.append(h('div',{class:'section-head'},h('h2',null,t('Video đã tải lên','Uploaded videos')),btn({v:'primary',icon:'plus',label:t('Nhập video','Import video'),onClick:()=>go('new')})));
  root.append(h('div',{class:'card'},emptyState('video',t('Danh sách video theo dự án','Videos are listed per project'),t('Video nguồn của mỗi dự án hiện tại được quản lý trong mục Dự án. Vào một dự án để xem lại video gốc.','Source videos are currently managed per project. Open a project to view its original video.'),false),h('div',{style:{textAlign:'center',paddingBottom:'26px'}},btn({v:'outline',label:t('Xem danh sách dự án','View project list'),onClick:()=>go('projects')})))); }
function pageOutputs(root){ root.append(h('div',{class:'section-head'},h('h2',null,t('Sản phẩm đã kết xuất','Rendered outputs'))));
  const done=S.projects.filter(p=>p.status==='COMPLETED');
  if(!done.length){ root.append(h('div',{class:'card'},emptyState('box',t('Chưa có sản phẩm nào','No outputs yet'),t('Sản phẩm (video/SRT/MP3) sẽ xuất hiện ở đây sau khi bạn kết xuất một dự án trong trình chỉnh sửa.','Outputs (video/SRT/MP3) will appear here after you render a project in the editor.'),false))); return; }
  const grid=h('div',{class:'proj-grid'}); done.forEach(p=>grid.append(projectCard(p))); root.append(grid); }
function pageAutopublish(root){ root.append(h('div',{class:'section-head'},h('h2',null,t('Tự động đăng bài','Auto-publish')),h('span',{class:'badge badge-warning'},t('Sắp ra mắt','Coming soon'))));
  root.append(h('div',{class:'card'},h('div',{style:{padding:'20px',display:'flex',flexDirection:'column',gap:'14px'}},h('p',{class:'hint',style:{margin:0,fontSize:'13px'}},t('Kết nối kênh YouTube, TikTok, Facebook để Lingvo tự đăng video đã dịch xong theo lịch bạn chọn.','Connect your YouTube, TikTok and Facebook channels so Lingvo can auto-publish finished videos on your schedule.')),
    ...['YouTube','TikTok','Facebook'].map(name=>h('div',{style:{display:'flex',alignItems:'center',gap:'10px',padding:'10px 12px',border:'1px solid var(--border)',borderRadius:'9px'}},h('div',{style:{width:32,height:32,borderRadius:8,background:'var(--muted)',display:'grid',placeItems:'center',fontWeight:800,fontSize:11}},name[0]),h('span',{class:'grow'},name),btn({v:'outline',label:t('Kết nối','Connect'),disabled:true})))))); }

/* =========================================================
   7) TRANG TẠO DỰ ÁN — link / upload / ngôn ngữ / đầu ra / nhà cung cấp / xác nhận
   ========================================================= */
const WIZ_STEPS=[t('Nguồn video','Source'),t('Ngôn ngữ','Languages'),t('Đầu ra','Output'),t('Nhà cung cấp','Providers'),t('Xác nhận','Confirm')];
let modelPickerCache = [];

// Chỉ hiển thị các model có trong danh sách người dùng đã chọn từ catalog.
// Tier là nhãn thực dụng để dễ chọn: Ưu tiên = dòng mạnh/đa năng; Cân bằng = chất lượng/tốc độ;
// Nhẹ = tiết kiệm tài nguyên; Thử nghiệm = model tên/nhánh thử nghiệm, không khẳng định benchmark tuyệt đối.
const MODEL_NAME_PATTERNS = [
  /^command a(?:\+| reasoning| vision| translate)?$/i, /^north mini code$/i, /^north small translate$/i,
  /^command r\+.*08-2024\)?$/i, /^sensenova 6\.[78] flash-lite$/i,
  /^big pickle.*$/i, /^space bunny alpha.*$/i, /^dots3-note preview.*$/i, /^ling 3\.0 flash sante.*$/i,
  /^qwen3\.(8|7|6) (omni flash|max|plus|flash|27b|max preview).*$/i,
  /^minimax m[23](?:\.[1257])?(?: highspeed)?.*$/i,
  /^mistral large 3$/i, /^mistral medium 3\.5$/i, /^mistral small 4$/i, /^codestral$/i, /^devstral 2$/i,
  /^ministral 3 (14b|8b|3b)$/i,
  /^deepseek v4(?:\.1 flash| flash vision| pro 0813| flash 0731| pro| flash).*$/i, /^deepseek v3\.[12]$/i,
];
function modelDisplayName(m){ return String(m.display_name||m.name||m.id||'').replace(/\s+/g,' ').trim(); }
function modelBaseName(m){ return modelDisplayName(m).replace(/\s*\(free\)\s*$/i,'').trim(); }
function isAllowedModel(m){ const n=modelBaseName(m); return MODEL_NAME_PATTERNS.some(rx=>rx.test(n)); }
function modelTier(m){
  if(m.quality_tier) return String(m.quality_tier);
  const n=modelBaseName(m).toLowerCase();
  if(/big pickle|space bunny alpha|dots3-note preview|ling 3\.0 flash sante/.test(n)) return 'experimental';
  if(/command a\+|command a reasoning|command a vision|command a$|qwen3\.8 (omni flash|max)|qwen3\.7 max|minimax m3|^minimax m2\.7(?: highspeed)?$|mistral large 3|mistral medium 3\.5|deepseek v4(?:\.1 flash| flash vision| pro)/.test(n)) return 'priority';
  if(/sensenova|ministral 3 (8b|3b)|deepseek v3\.1/.test(n)) return 'light';
  return 'balanced';
}
function modelTierMeta(tier){
  const map={
    priority:{vi:'🔥 Ưu tiên',en:'🔥 Priority'},
    balanced:{vi:'⚡ Cân bằng',en:'⚡ Balanced'},
    light:{vi:'🪶 Nhẹ',en:'🪶 Light'},
    experimental:{vi:'🧪 Thử nghiệm',en:'🧪 Experimental'}
  };
  return map[tier]||map.balanced;
}
function modelTierRank(tier){ return ({priority:0,balanced:1,light:2,experimental:3}[tier]??9); }
function stageModelCompatible(m,stage){
  const mod=String(m.modality||'chat').toLowerCase();
  const cap=m.capabilities||{};
  if(stage==='ocr') return mod==='ocr' || !!cap.vision || /vision|omni/i.test(modelDisplayName(m));
  if(stage==='stt') return mod==='stt' || !!cap.audio || /omni|audio|speech|transcrib|whisper/i.test(modelDisplayName(m)+' '+String(m.id||''));
  if(stage==='tts') return mod==='tts' || !!cap.tts;
  if(stage==='translate') return mod==='chat' || !m.modality;
  return true;
}
async function openModelPicker(stage){
  const w=S.wiz;
  let models=[];
  try{ models=await api.getModels('all'); }
  catch(e){ toast(e.message||t('Không tải được catalog model','Could not load model catalog'),'error'); return; }
  models=(Array.isArray(models)?models:[]).filter(isAllowedModel);
  const compatible=models.filter(m=>stageModelCompatible(m,stage));
  // Nếu catalog không khai báo modality/capability đáng tin cậy, vẫn cho phép các model đã whitelist
  // để mục "Full" không bị mất model; riêng OCR vẫn ưu tiên Vision/Omni.
  const list=(stage==='full'?models:compatible.length?compatible:models).sort((a,b)=>{
    const tr=modelTierRank(modelTier(a))-modelTierRank(modelTier(b));
    return tr || modelDisplayName(a).localeCompare(modelDisplayName(b),'en');
  });
  const overlay=h('div',{style:{position:'fixed',inset:0,zIndex:500,background:'rgba(0,0,0,.72)',display:'grid',placeItems:'center',padding:'20px'}},
    h('div',{class:'card',style:{width:'min(860px,96vw)',maxHeight:'84vh',overflow:'hidden',display:'flex',flexDirection:'column',background:'var(--card)',border:'1px solid var(--border)'}},
      h('div',{style:{padding:'16px 18px',display:'flex',alignItems:'center',gap:'10px',borderBottom:'1px solid var(--border)'}},
        h('div',{class:'grow'},
          h('b',null,t(`Chọn mô hình ${stage.toUpperCase()}`,'Choose '+stage.toUpperCase()+' model')),
          h('div',{class:'hint'},t('Chỉ hiển thị model đã chọn · 🔥 ưu tiên · ⚡ cân bằng · 🪶 nhẹ · 🧪 thử nghiệm','Only selected models · 🔥 priority · ⚡ balanced · 🪶 light · 🧪 experimental'))),
        btn({icon:'x',onClick:()=>overlay.remove()})),
      h('div',{style:{padding:'12px 16px',overflow:'auto',display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(250px,1fr))',gap:'8px'}},
        list.length?list.map(m=>{
          const tier=modelTier(m), meta=modelTierMeta(tier), selected=getSelectedModel(w,stage)===m.id;
          return h('button',{type:'button',class:'provider-card','data-on':selected,onClick:()=>{setSelectedModel(w,stage,m.id);overlay.remove();renderPage();}},
            h('span',{class:'rd'}),
            h('span',{class:'grow'},
              h('div',{style:{display:'flex',alignItems:'center',gap:'7px',flexWrap:'wrap'}},h('b',null,modelDisplayName(m)),h('span',{style:{fontSize:'10px',fontWeight:'800',padding:'2px 6px',borderRadius:'999px',border:'1px solid var(--border)',whiteSpace:'nowrap'}},lang==='vi'?meta.vi:meta.en)),
              h('small',{class:'muted'},`${m.id||''}${m.access_tier?' · '+m.access_tier:''}`)));
        }):
          [h('div',{class:'config-note',style:{gridColumn:'1/-1'}},t('Catalog không có model phù hợp. Model OCR cần Vision/Omni; STT/TTS chỉ hiện model có capability tương ứng khi gateway khai báo.','No compatible model. OCR needs Vision/Omni; STT/TTS only show models with the corresponding gateway capability.'))]),
      h('div',{style:{padding:'12px 16px',borderTop:'1px solid var(--border)',display:'flex',justifyContent:'space-between',gap:'8px'}},
        h('span',{class:'hint'},`${list.length} ${t('model đã chọn','allowlisted models')}`),btn({v:'outline',label:t('Đóng','Close'),onClick:()=>overlay.remove()}))));
  document.body.append(overlay);
}
function getSelectedModel(w,stage){return stage==='stt'?w.providers.sttModel:stage==='ocr'?w.providers.ocrModel:stage==='translate'?w.providers.translateModel:stage==='tts'?w.providers.ttsModel:w.providers.fullModel;}
function setSelectedModel(w,stage,id){
  if(stage==='stt') w.providers.sttModel=id;
  else if(stage==='ocr'){w.providers.ocrModel=id;}
  else if(stage==='translate') w.providers.translateModel=id;
  else if(stage==='tts') w.providers.ttsModel=id;
  else w.providers.fullModel=id;
}
function modelButton(stage,label,model){
  return h('button',{type:'button',class:'config-output-option','data-on':!!model,onClick:()=>openModelPicker(stage)},
    h('b',null,label),h('div',{class:'hint'},model||t('Tự động / provider mặc định','Automatic / provider default')));
}

function pageNewProject(root){
  const w=S.wiz, keys=(S.settings&&S.settings.keys)||{};
  const wrap=h('div',{class:'config-page'});
  const left=h('div',{class:'card config-video-card'});
  const right=h('div',{class:'card config-panel-card'});

  // LEFT: video/source, giống bố cục ảnh mẫu.
  const tabs=h('div',{class:'config-source-tabs'},
    btn({v:w.source==='link'?'secondary':'outline',icon:'link',label:t('Nhập link','Import link'),onClick:()=>{w.source='link';renderPage();}}),
    btn({v:w.source==='upload'?'secondary':'outline',icon:'upload',label:t('Tải video','Upload video'),onClick:()=>{w.source='upload';renderPage();}}));
  left.append(tabs);
  if(!w.video){
    if(w.source==='link'){
      const input=h('input',{class:'input',placeholder:t('Dán link YouTube, TikTok, Douyin…','Paste a video URL…'),value:w.url,onInput:e=>{w.url=e.target.value;}});
      left.append(h('div',{class:'field'},h('div',{class:'lbl'},t('Nguồn video','Video source')),h('div',{style:{display:'flex',gap:'8px'}},input,btn({v:'primary',icon:'search',label:t('Phân tích','Analyze'),busy:w.urlBusy,disabled:w.urlBusy||!w.url.trim(),onClick:analyzeUrl}))));
    } else {
      const input=h('input',{type:'file',accept:'video/*',hidden:true,onChange:e=>{const f=e.target.files?.[0];if(f)uploadFile(f);}});
      left.append(h('label',{class:'drop-zone',style:{minHeight:'330px'},ondragover:e=>e.preventDefault(),ondrop:e=>{e.preventDefault();const f=e.dataTransfer.files?.[0];if(f)uploadFile(f);}},input,h('div',{class:'dic'},ic('upload',26)),h('b',null,t('Kéo thả video vào đây hoặc bấm để chọn file','Drop a video here or click to choose')),h('span',{class:'hint'},'MP4, MOV, WEBM · '+t('tối đa 2GB','up to 2GB'))));
    }
  } else {
    // Khung video kiểu editor: chỉ hiển thị video + nút play phủ lên (không dùng controls gốc).
    const videoEl=h('video',{class:'config-video',preload:'metadata',playsinline:true});
    if(w.video._file){ try{videoEl.src=URL.createObjectURL(w.video._file);}catch(e){} } else if(w.video._url) { videoEl.src=w.video._url; }
    else if(w.video.thumbnail){ videoEl.poster=w.video.thumbnail; }
    const videoStage=h('div',{class:'config-video-stage'});
    const playLayer=h('div',{class:'config-play-layer',onClick:()=>{ if(videoEl.paused) videoEl.play().catch(()=>{}); else videoEl.pause(); }});
    const renderPlay=()=>{ playLayer.innerHTML=''; if(videoEl.paused) playLayer.append(h('div',{class:'config-play-btn'},(()=>{const i=ic('play',26);i.style.marginLeft='3px';return i;})())); };
    ['play','pause','ended'].forEach(ev=>videoEl.addEventListener(ev,renderPlay));
    // Vùng nội dung video thật (tránh lệch khi video không đúng tỷ lệ 16:9)
    const contentBox=()=>{ const r=videoStage.getBoundingClientRect(); const vw=videoEl.videoWidth||16, vh=videoEl.videoHeight||9; const sc=Math.min(r.width/vw,r.height/vh); if(!isFinite(sc)||sc<=0) return null; const cw=vw*sc, ch=vh*sc; return {left:(r.width-cw)/2,top:(r.height-ch)/2,width:cw,height:ch}; };
    let drawOcrRect=null;
    videoEl.addEventListener('loadedmetadata',()=>{ if(drawOcrRect) drawOcrRect(); });
    videoStage.append(videoEl,playLayer); renderPlay();
    // KHUNG QUÉT PHỤ ĐỀ OCR — chỉ hiển thị khi chọn OCR, chỉ ở màn hình cấu hình này.
    if(w.subtitleSource==='ocr'){
      const rect=h('div',{class:'ocr-video-rect'},h('span',{class:'ocr-tag'},'OCR'));
      const draw=()=>{ const b=contentBox(); const r=w.ocrRegion; if(!b){ Object.assign(rect.style,{left:r.x+'%',top:r.y+'%',width:r.w+'%',height:r.h+'%'}); return; } Object.assign(rect.style,{left:(b.left+r.x/100*b.width)+'px',top:(b.top+r.y/100*b.height)+'px',width:(r.w/100*b.width)+'px',height:(r.h/100*b.height)+'px'}); };
      drawOcrRect=draw;
      const startDrag=(e,mode)=>{ e.preventDefault(); e.stopPropagation(); mode=mode==='l'?'w':mode==='r'?'e':mode; const b=contentBox(); if(!b) return; const x0=e.clientX,y0=e.clientY,init=Object.assign({},w.ocrRegion);
        const mv=ev=>{ const dx=(ev.clientX-x0)/b.width*100, dy=(ev.clientY-y0)/b.height*100; let {x,y,w:rw,h:rh}=init;
          if(mode==='move'){x=clamp(x+dx,0,100-rw);y=clamp(y+dy,0,100-rh);}
          else { if(mode.includes('e')) rw=clamp(rw+dx,1,100-x); if(mode.includes('w')){const nx=clamp(x+dx,0,x+rw-1);rw+=x-nx;x=nx;} if(mode.includes('s')) rh=clamp(rh+dy,1,100-y); if(mode.includes('n')){const ny=clamp(y+dy,0,y+rh-1);rh+=y-ny;y=ny;} }
          w.ocrRegion={x:+x.toFixed(2),y:+y.toFixed(2),w:+rw.toFixed(2),h:+rh.toFixed(2)}; draw(); };
        const up=()=>{window.removeEventListener('pointermove',mv);window.removeEventListener('pointerup',up);};
        window.addEventListener('pointermove',mv); window.addEventListener('pointerup',up); };
      rect.addEventListener('pointerdown',e=>{ if(e.target===rect) startDrag(e,'move'); });
      ['n','s','l','r','nw','ne','sw','se'].forEach(mode=>{ rect.append(h('i',{'data-side':mode,onPointerdown:e=>startDrag(e,mode)})); });
      videoStage.append(rect); draw();
      new ResizeObserver(()=>draw()).observe(videoStage);
    }
    left.append(videoStage,
      w.subtitleSource==='ocr'?h('div',{class:'hint',style:{marginTop:'8px'}},t('Kéo khung để dời vùng quét phụ đề; kéo các nút tròn ở cạnh/góc để thay đổi kích thước vùng.','Drag the frame to move the subtitle scan region; drag the round handles on edges/corners to resize it.')):null,
      h('div',{style:{display:'flex',justifyContent:'space-between',alignItems:'center',marginTop:'10px'}},h('div',null,h('b',null,w.video.name||t('Video đã chọn','Selected video')),h('div',{class:'hint'},`${Number(w.video.duration_minutes||0).toFixed(1)} ${t('phút','min')}`)),btn({v:'outline',icon:'x',label:t('Chọn video khác','Choose another'),onClick:()=>{w.video=null;renderPage();}})));
  }
  left.append(h('div',{class:'config-note',style:{marginTop:'12px'}},t('Tất cả video, transcript, audio TTS, cache, project và output sẽ được lưu dưới DATA_ROOT trên ổ D.','All video, transcript, TTS audio, cache, project and output files are stored below DATA_ROOT on drive D.')));

  // RIGHT: tất cả cấu hình trên một màn hình, không còn phải đi qua 5 bước.
  right.append(h('div',{class:'config-section'},h('div',{class:'config-section-title'},ic('languages',14),t('Ngôn ngữ','Languages')),h('div',{class:'config-two'},
    h('div',{class:'field'},h('div',{class:'lbl'},t('Nguồn','Source')),sel(Object.keys(LANGS).map(c=>[c,`${L(c)[2]} ${L(c)[lang==='vi'?0:1]}`]),w.srcLang,v=>{w.srcLang=v;renderPage();})),
    h('div',{class:'field'},h('div',{class:'lbl'},t('Đầu ra','Target')),sel(Object.keys(LANGS).filter(c=>c!=='auto').map(c=>[c,`${L(c)[2]} ${L(c)[lang==='vi'?0:1]}`]),w.tgtLang,v=>{w.tgtLang=v;renderPage();})))));

  right.append(h('div',{class:'config-section'},h('div',{class:'config-section-title'},ic('captions',14),t('Nguồn phụ đề','Subtitle source')),h('div',{class:'config-output-grid'},
    h('button',{type:'button',class:'config-output-option','data-on':w.subtitleSource==='stt',onClick:()=>{w.subtitleSource='stt';renderPage();}},h('b',null,t('Giọng nói (STT)','Speech (STT)')),h('div',{class:'hint'},t('Nhận dạng lời thoại','Recognize speech'))),
    h('button',{type:'button',class:'config-output-option','data-on':w.subtitleSource==='ocr',onClick:()=>{w.subtitleSource='ocr';renderPage();}},h('b',null,t('Phụ đề cứng (OCR)','Hardsub (OCR)')),h('div',{class:'hint'},t('Đọc chữ trên video','Read burned-in subtitles'))))));

  right.append(h('div',{class:'config-section'},h('div',{class:'config-section-title'},ic('film',14),t('Đầu ra','Output')),h('div',{class:'config-output-grid'},
    h('button',{type:'button',class:'config-output-option','data-on':!w.wantDubbing,onClick:()=>{w.wantDubbing=false;renderPage();}},h('b',null,t('Chỉ phụ đề','Subtitles only')),h('div',{class:'hint'},t('Không tạo TTS','Do not generate dubbing'))),
    h('button',{type:'button',class:'config-output-option','data-on':w.wantDubbing,onClick:()=>{w.wantDubbing=true;renderPage();}},h('b',null,t('Phụ đề + lồng tiếng','Subtitles + dubbing')),h('div',{class:'hint'},t('Tạo TTS trong editor','Generate TTS in editor'))))));

  const makeProviderSection=(title,icon,arr,key,on)=>h('div',{class:'config-section'},h('div',{class:'config-section-title'},ic(icon,14),title),h('div',{class:'config-provider-list'},arr.map(pr=>providerCard(pr,w.providers[key],v=>{w.providers[key]=v;renderPage();},keys))));
  if(w.subtitleSource==='ocr') right.append(makeProviderSection(t('Nhận dạng phụ đề','Subtitle OCR'),'captions',OCR_PROVIDERS,'ocr'));
  if(w.subtitleSource!=='ocr') right.append(makeProviderSection(t('Nhận dạng giọng nói','Speech-to-text'),'mic',STT_PROVIDERS,'stt'));
  right.append(makeProviderSection(t('Dịch thuật','Translation'),'languages',TX_PROVIDERS,'translate'));
  if(w.wantDubbing) right.append(makeProviderSection(t('Lồng tiếng','Text-to-speech'),'volume',TTS_PROVIDERS,'tts'));

  right.append(h('div',{class:'config-section'},h('div',{class:'config-section-title'},ic('sparkles',14),t('Mô hình AI riêng theo mục','Per-stage AI models')),
    h('div',{class:'config-output-grid'},
      w.subtitleSource==='ocr'?modelButton('ocr',t('Model OCR','OCR model'),w.providers.ocrModel||'PP-OCRv5-mobile'):modelButton('stt',t('Model STT','STT model'),w.providers.sttModel),
      modelButton('translate',t('Model dịch','Translation model'),w.providers.translateModel),
      w.wantDubbing?modelButton('tts',t('Model TTS','TTS model'),w.providers.ttsModel):null,
      modelButton('full',t('Full toàn bộ model','All models'),w.providers.fullModel)
    )));

  if(w.wantDubbing && w.providers.tts==='xkira_tts'){
    const voiceBody=h('div',{class:'config-voice-grid'});
    if(XKIRA_VOICES.length) voiceBody.append(...XKIRA_VOICES.slice(0,4).map(v=>h('button',{type:'button',class:'opt-card','data-on':(w.providers.xkiraVoice||w.providers.voice)===v.id,onClick:()=>{w.providers.xkiraVoice=v.id;w.providers.voice=v.id;renderPage();}},h('b',null,v.name||v.id),h('small',{class:'muted'},v.gender==='female'?t('Nữ','Female'):v.gender==='male'?t('Nam','Male'):''))));
    else voiceBody.append(h('div',{class:'config-note'},t('Chưa lấy được catalog xKiro. Backend sẽ lấy tối đa 4 giọng thật từ API khi có kết nối.','xKiro catalog is not available yet. The backend will load up to 4 real API voices when connected.')));
    right.append(h('div',{class:'config-section'},h('div',{class:'config-section-title'},ic('volume',14),t('Giọng đọc xKiro','xKiro voice')),voiceBody));
  }
  right.append(h('div',{class:'config-actions'},btn({v:'outline',icon:'arrowLeft',label:t('Quay lại','Back'),onClick:()=>go('dashboard')}),btn({v:'primary',icon:'sparkles',label:t('Bắt đầu xử lý','Start processing'),busy:w.creating,disabled:w.creating||!w.video,onClick:createProjectFromWizard})));
  wrap.append(left,right); root.append(wrap);
}
function canAdvance(w){ if(w.step===0) return !!w.video; return true; }
function stepSource(body){ const w=S.wiz;
  const tabs=h('div',{class:'source-tabs'},h('button',{type:'button','data-on':w.source==='link',onClick:()=>{w.source='link';renderPage();}},ic('link',14),t('Nhập link','Import link')),h('button',{type:'button','data-on':w.source==='upload',onClick:()=>{w.source='upload';renderPage();}},ic('upload',14),t('Tải lên','Upload'))); body.append(tabs);
  const wrap=h('div',{style:{marginTop:'16px',display:'flex',flexDirection:'column',gap:'14px'}}); body.append(wrap);
  if(w.source==='link'){ wrap.append(h('div',{class:'platform-grid'},PLATFORMS.map(([id,name,color])=>h('div',{class:'platform-chip'},h('b',{style:{background:color}},name[0]),name))));
    const input=h('input',{class:'input',placeholder:t('Dán link YouTube, TikTok, Douyin…','Paste a YouTube, TikTok, Douyin link…'),value:w.url,onInput:e=>{w.url=e.target.value;}});
    wrap.append(h('div',{class:'field'},h('div',{class:'lbl'},t('Đường dẫn video','Video URL')),h('div',{style:{display:'flex',gap:'8px'}},input,btn({v:'primary',sm:false,icon:'search',label:t('Phân tích','Analyze'),busy:w.urlBusy,disabled:w.urlBusy||!w.url.trim(),onClick:analyzeUrl}))));
    if(w.video) wrap.append(videoPreview(w.video,()=>{w.video=null;renderPage();})); }
  else { if(!w.video){ const input=h('input',{type:'file',accept:'video/*',hidden:true,onChange:e=>{ const f=e.target.files&&e.target.files[0]; if(f) uploadFile(f); }}); wrap.append(h('label',{class:'drop-zone',ondragover:e=>e.preventDefault(),ondrop:e=>{e.preventDefault(); const f=e.dataTransfer.files&&e.dataTransfer.files[0]; if(f) uploadFile(f);}},input,h('div',{class:'dic'},ic('upload',20)),h('b',null,t('Kéo thả video vào đây hoặc bấm để chọn file','Drag & drop a video here or click to browse')),h('span',{class:'hint'},'MP4, MOV, WEBM · '+t('tối đa 2GB','up to 2GB')))); if(w.uploadBusy) wrap.append(h('div',{class:'field'},h('div',{class:'lbl'},t('Đang tải lên…','Uploading…')+' '+w.uploadProgress+'%'),h('div',{class:'progress'},h('i',{style:{width:w.uploadProgress+'%'}})))); }
    else wrap.append(videoPreview(w.video,()=>{w.video=null;renderPage();})); } }
function videoPreview(v,onReset){ return h('div',{class:'preview-card'}, v.thumbnail?h('img',{src:v.thumbnail}):h('div',{style:{width:96,height:60,borderRadius:7,background:'var(--muted)',display:'grid',placeItems:'center',flexShrink:0}},ic('film',18)), h('div',{class:'grow'},h('b',null,v.name),h('span',null,`${v.duration_minutes.toFixed(1)} ${t('phút','min')}`)), btn({v:'ghost',icon:'x',title:t('Chọn video khác','Choose a different video'),onClick:onReset})); }
async function analyzeUrl(){ const w=S.wiz; w.urlBusy=true; renderPage(); try{ const v=await api.importUrl(w.url.trim()); w.video=v; }catch(e){ toast(e.message||t('Không thể phân tích link này','Could not analyze this link'),'error'); } finally{ w.urlBusy=false; renderPage(); } }
async function uploadFile(file){ const w=S.wiz; w.uploadBusy=true; w.uploadProgress=0; renderPage(); try{ const v=await api.uploadVideo(file,p=>{w.uploadProgress=p; const bar=$('.progress i',$('#contentInner')); if(bar) bar.style.width=p+'%';}); w.video=v; }catch(e){ toast(e.message||t('Tải video thất bại','Upload failed'),'error'); } finally{ w.uploadBusy=false; renderPage(); } }
function stepLanguages(body){ const w=S.wiz; const opts=Object.keys(LANGS).map(c=>[c,`${L(c)[2]} ${L(c)[lang==='vi'?0:1]}`]);
  body.append(h('div',{class:'lang-grid'},h('div',{class:'field'},h('div',{class:'lbl'},t('Ngôn ngữ gốc','Source language')),sel(opts,w.srcLang,v=>{w.srcLang=v;})),h('div',{class:'mid'},ic('arrowRight',18)),h('div',{class:'field'},h('div',{class:'lbl'},t('Ngôn ngữ đích','Target language')),sel(opts.filter(o=>o[0]!=='auto'),w.tgtLang,v=>{w.tgtLang=v;}))));
  body.append(h('div',{class:'hint',style:{marginTop:'14px'}},t('Chọn "Tự động" nếu bạn không chắc video đang nói ngôn ngữ gì — AI sẽ tự nhận diện.','Choose "Auto" if you are not sure which language the video speaks — the AI will detect it.'))); }
function stepOutput(body){ const w=S.wiz;
  body.append(h('div',{class:'field'},h('div',{class:'lbl'},t('Nguồn phụ đề','Subtitle source')),h('div',{class:'option-grid'},
    h('button',{type:'button',class:'opt-card','data-on':w.subtitleSource==='stt',onClick:()=>{w.subtitleSource='stt';renderPage();}},h('span',{class:'oic'},ic('mic',16)),h('span',null,h('b',null,t('Giọng nói (STT)','Speech (STT)')),h('small',null,t('Nhận dạng lời thoại bằng AI','AI speech recognition')))),
    h('button',{type:'button',class:'opt-card','data-on':w.subtitleSource==='ocr',onClick:()=>{w.subtitleSource='ocr';renderPage();}},h('span',{class:'oic'},ic('captions',16)),h('span',null,h('b',null,t('Phụ đề cứng (OCR)','Hardsub (OCR)')),h('small',null,t('Đọc phụ đề có sẵn trên video','Read subtitles burned into the video')))))));
  body.append(h('div',{class:'field',style:{marginTop:'16px'}},h('div',{style:{display:'flex',alignItems:'center',justifyContent:'space-between'}},h('div',null,h('div',{class:'lbl'},t('Lồng tiếng AI','AI dubbing')),h('div',{class:'hint'},t('Tạo giọng đọc AI cho ngôn ngữ đích','Generate an AI voice for the target language'))),sw(w.wantDubbing,v=>{w.wantDubbing=v;renderPage();}))));
  if(w.wantDubbing){ const vf=h('div',{class:'field',style:{marginTop:'12px'}},h('div',{class:'lbl'},t('Giọng đọc mặc định','Default voice'))); const opts=[]; if(w.providers.tts==='xkira_tts'){ XKIRA_VOICES.slice(0,4).forEach(v=>opts.push([v.id,`xKiro · ${v.name}`])); } else { voicesFor(w.tgtLang).slice(0,2).forEach(([id,n])=>opts.push([id,`Microsoft Edge · ${n}`])); } if(opts.length){ const ss=sel(opts,w.providers.xkiraVoice||w.providers.voice||opts[0][0],v=>{w.providers.voice=v;if(w.providers.tts==='xkira_tts')w.providers.xkiraVoice=v;renderPage();},{style:'height:36px'}); const pv=btn({v:'primary',icon:'volume',label:t('Nghe thử','Preview'),onClick:async()=>{try{const u=await api.previewVoice(ss.value,w.providers.tts==='xkira_tts'?'xkira_tts':'edge_tts',w.tgtLang);const au=new Audio(u);au.onended=au.onerror=()=>URL.revokeObjectURL(u);au.play();}catch(e){toast(e.message,'error');}}}); vf.append(h('div',{style:{display:'flex',gap:'7px'}},ss,pv)); } body.append(vf); }
  if(w.subtitleSource!=='ocr') body.append(h('div',{style:{display:'flex',alignItems:'center',justifyContent:'space-between',marginTop:'16px'}},h('div',null,h('div',{class:'lbl'},t('Lọc đoạn transcript nhiễu','Filter noisy transcript segments')),h('div',{class:'hint'},t('Loại bỏ câu rời rạc, không mạch lạc — xử lý lâu hơn một chút','Drop fragmented, incoherent lines — slightly slower'))),sw(w.filterNoise,v=>{w.filterNoise=v;})));
  else body.append(h('div',{class:'config-section',style:{marginTop:'14px'}},
    h('div',{class:'config-section-title'},ic('captions',13),t('OCR thích nghi · Khóa timeline','Adaptive OCR · Frame-locked timeline')),
    h('div',{class:'config-note'},t('OCR tự học vị trí và kiểu phụ đề bên trong ROI, theo dõi chữ qua nhiều frame, loại watermark/chữ nền theo tính ổn định và khóa start/end theo frame thật. Không dùng STT để tạo nội dung OCR.','OCR adapts to subtitle position and appearance inside the ROI, tracks text across frames, filters persistent watermark/background text, and locks start/end to actual video frames. STT is not used to create OCR text.')),
    h('div',{style:{display:'flex',gap:'6px',flexWrap:'wrap',marginTop:'8px'}},
      h('span',{class:'badge badge-cyan'},'Adaptive position'),
      h('span',{class:'badge badge-cyan'},'Multi-frame verify'),
      h('span',{class:'badge badge-success'},'Frame locked'),
      h('span',{class:'badge badge-neutral'},'CPU target ≤70%'))
  ));
}
function stepProviders(body){ const w=S.wiz; const keys=(S.settings&&S.settings.keys)||{};
  if(w.subtitleSource==='ocr') body.append(h('div',{class:'config-section'},h('div',{class:'config-section-title'},ic('captions',13),t('Nhận diện phụ đề gốc','Burned-in subtitle OCR')),h('div',{class:'config-note'},t('OCR chỉ đọc hình ảnh trong ROI đã chọn; không phân tích âm thanh, không chạy STT. Bộ lọc loại các ký tự Latin đơn lẻ kiểu A B C D V B N M thường đến từ giao diện/môi trường. Timeline được khóa theo frame video, không lấy thời gian từ âm thanh.','OCR reads image pixels only inside the selected ROI; it does not analyze audio or run STT. A filter rejects isolated Latin glyphs such as A B C D V B N M commonly picked up from UI/environment. Timeline is locked to video frames, never audio timing.'))));
  else body.append(h('div',null,h('div',{class:'lbl',style:{marginBottom:'8px'}},t('Nhận dạng giọng nói','Speech-to-Text')),h('div',{style:{display:'flex',flexDirection:'column',gap:'6px'}},STT_PROVIDERS.map(pr=>providerCard(pr,w.providers.stt,v=>{w.providers.stt=v;renderPage();},keys)))));
  body.append(h('div',{class:'hint',style:{margin:'16px 0 8px'}}));
  body.append(h('div',null,h('div',{class:'lbl',style:{marginBottom:'8px'}},t('Dịch thuật','Translation')),h('div',{style:{display:'flex',flexDirection:'column',gap:'6px'}},TX_PROVIDERS.map(pr=>providerCard(pr,w.providers.translate,v=>{w.providers.translate=v;renderPage();},keys)))));
  if(w.wantDubbing){ body.append(h('div',{style:{margin:'16px 0 8px'}})); body.append(h('div',null,h('div',{class:'lbl',style:{marginBottom:'8px'}},t('Lồng tiếng','Text-to-Speech')),h('div',{style:{display:'flex',flexDirection:'column',gap:'6px'}},TTS_PROVIDERS.map(pr=>providerCard(pr,w.providers.tts,v=>{w.providers.tts=v;renderPage();},keys))))); }
  body.append(h('div',{style:{marginTop:'16px',display:'flex',alignItems:'center',justifyContent:'space-between'}},h('div',null,h('div',{class:'lbl'},t('Soát lại bản dịch','Refine translations')),h('div',{class:'hint'},t('AI đọc lại bản dịch lần hai để mượt hơn — dịch chậm hơn khoảng gấp đôi','A second AI pass smooths the wording — roughly doubles translation time'))),sw(w.providers.refine,v=>{w.providers.refine=v;}))); }
function stepConfirm(body){ const w=S.wiz; const row=(k,v)=>h('div',{class:'summary-row'},h('span',null,k),h('b',null,v));
  body.append(h('div',null, w.video?videoPreview(w.video,()=>{}):null, h('div',{style:{marginTop:'14px'}},
    row(t('Ngôn ngữ','Languages'),`${L(w.srcLang)[2]} ${L(w.srcLang)[lang==='vi'?0:1]} → ${L(w.tgtLang)[2]} ${L(w.tgtLang)[lang==='vi'?0:1]}`),
    row(t('Nguồn phụ đề','Subtitle source'),w.subtitleSource==='ocr'?t('Phụ đề cứng (OCR)','Hardsub (OCR)'):t('Giọng nói (STT)','Speech (STT)')),
    row(t('Lồng tiếng AI','AI dubbing'),w.wantDubbing?t('Bật','On'):t('Tắt','Off')),
    row(t('Nhận dạng','Recognition'),w.subtitleSource==='ocr'?t('Adaptive OCR · PP-OCRv5 mobile','Adaptive OCR · PP-OCRv5 mobile'):(STT_PROVIDERS.find(p=>p.id===w.providers.stt)||{}).name),
    row(t('Dịch thuật','Translation'),(TX_PROVIDERS.find(p=>p.id===w.providers.translate)||{}).name),
    w.wantDubbing?row(t('Giọng lồng tiếng','TTS provider'),(TTS_PROVIDERS.find(p=>p.id===w.providers.tts)||{}).name):null)));
  body.append(h('div',{class:'hint',style:{marginTop:'14px'}},t('Sau khi tạo, dự án sẽ được phân tích và tách giọng tự động. Bạn sẽ thấy màn hình tiến trình cho đến khi sẵn sàng chỉnh sửa.','Once created, the project will be analyzed and voices separated automatically. You will see a progress screen until it is ready to edit.'))); }
async function createProjectFromWizard(){ const w=S.wiz; if(!w.video){ toast(t('Vui lòng chọn video trước','Please choose a video first'),'warning'); w.step=0; renderPage(); return; } w.creating=true; renderPage();
  try{ await api.saveProviders({ stt_provider:w.providers.stt, ocr_provider:w.providers.ocr, translate_provider:w.providers.translate, tts_provider:w.providers.tts, stt_model:w.providers.sttModel, ocr_model:w.providers.ocrModel, translate_model:w.providers.translateModel, tts_model:w.providers.ttsModel, full_model:w.providers.fullModel, translate_refine:w.providers.refine, edge_voice:w.providers.tts==='edge_tts'?w.providers.voice:'', xkira_voice:w.providers.tts==='xkira_tts'?(w.providers.xkiraVoice||w.providers.voice):'' });
    const p=await api.createProject({
      video_name:w.video.name, source_lang:w.srcLang, target_lang:w.tgtLang,
      subtitle_source:w.subtitleSource, want_subtitle:true, want_dubbing:w.wantDubbing,
      filter_transcript_noise:w.filterNoise, ocr_region:w.ocrRegion,
      stt_provider:w.providers.stt, translate_provider:w.providers.translate,
      tts_provider:w.providers.tts, stt_model:w.providers.sttModel, ocr_model:w.providers.ocrModel||'PP-OCRv5-mobile', translate_model:w.providers.translateModel, tts_model:w.providers.ttsModel, full_model:w.providers.fullModel, ocr_provider:w.subtitleSource==='ocr'?'local':'local', ocr_mode:'HIGH_RECALL', voice:w.providers.xkiraVoice||w.providers.voice, translate_refine:w.providers.refine,
      file:w.video._file, url:w.video._url, import_id:w.video._import_id
    });
    S.projects.unshift(p); if(api===demoApiInstance) await api.startTranscribe(p.id); toast(w.subtitleSource==='ocr'?t('Đã tạo dự án — mở trình chỉnh sửa và OCR đang chạy trực tiếp.','Project created — opening the editor while OCR runs live.'):t('Đã tạo dự án, đang phân tích video…','Project created, analyzing the video…'),'success'); if(w.subtitleSource==='ocr'){ store.set('lingvo_auto_ocr_project',p.id); notifyParent('lingvo:open-editor',{projectId:p.id,autoOCR:true}); } else go('processing',{projectId:p.id});
  }catch(e){ toast(e.message||t('Không thể tạo dự án','Could not create the project'),'error'); } finally{ w.creating=false; } }

/* =========================================================
   8) TRANG XỬ LÝ (LOADING) — theo dõi tiến trình rồi mở Editor
   ========================================================= */
const STAGE_META={ QUEUING:[t('Đang chờ trong hàng đợi','Waiting in queue'),'clock'], EXTRACT_AUDIO:[t('Tách âm thanh','Extract audio'),'volume'], STT:[t('Nhận dạng giọng nói','Transcribe'),'mic'], OCR:[t('Chỉ đọc phụ đề từ hình ảnh (OCR)','Read subtitles from video frames only'),'captions'] };
let procTimer=null;
function pageProcessing(root){ if(procTimer){clearInterval(procTimer);procTimer=null;} const id=S.processingId; const p=S.projects.find(x=>x.id===id);
  const wrap=h('div',{class:'proc-wrap'}); root.append(wrap);
  if(!p){ wrap.append(emptyState('alert',t('Không tìm thấy dự án đang xử lý','No processing project found'),null,false)); return; }
  const stages=p.subtitleSource==='ocr'?['QUEUING','OCR']:['QUEUING','EXTRACT_AUDIO','STT'];
  const renderCard=()=>{ wrap.innerHTML=''; const done=p.status!=='PROCESSING'&&p.status!=='QUEUED'; const failed=p.status==='FAILED';
    const ai=Math.max(0,stages.indexOf(p.stage||'QUEUING'));
    const card=h('div',{class:'card proc'}, h('div',{class:'orb'},ic(done?'check':failed?'alert':(STAGE_META[p.stage]||STAGE_META.QUEUING)[1],38,!done&&!failed?'spin-slow':'')),
      h('div',null,h('div',{style:{fontWeight:700,fontSize:'18px',marginBottom:'6px'}},done?t('Video đã sẵn sàng!','Your video is ready!'):failed?t('Xử lý thất bại','Processing failed'):(p.stage==='QUEUING'?(p.queuePosition?t(`Đang chờ trong hàng đợi… (vị trí #${p.queuePosition})`,`Waiting in queue… (#${p.queuePosition})`):t('Đang chờ trong hàng đợi…','Waiting in queue…')):(STAGE_META[p.stage]||STAGE_META.QUEUING)[0])),h('div',{class:'muted',style:{fontSize:'13px'}},p.name)));
    if(!done) card.append(h('div',{style:{width:'100%'}},h('div',{class:'progress'},h('i',{style:{width:p.progress+'%'}})),h('div',{class:'mono muted',style:{marginTop:'8px',fontSize:'12px',textAlign:'right'}},p.progress+'%')));
    card.append(h('div',{class:'stage-list'},stages.map((s,i)=>h('div',{class:'stage-item','data-s':done||i<ai?'done':i===ai?'active':''},ic(done||i<ai?'check':(STAGE_META[s]||[])[1],15,i===ai&&!done?'spin':''),(STAGE_META[s]||[''])[0]))));
    if(done) card.append(btn({v:'primary',sm:false,icon:'sparkles',label:t('Mở trình chỉnh sửa','Open editor'),cls:'btn-block',onClick:()=>notifyParent('lingvo:open-editor',{projectId:p.id})}));
    else card.append(btn({v:'outline',sm:false,label:t('Về bảng điều khiển (tiếp tục ở nền)','Back to dashboard (keeps running)'),cls:'btn-block',onClick:()=>go('dashboard')}));
    wrap.append(card); };
  renderCard();
  procTimer=setInterval(async()=>{ try{ const fresh=await api.getProject(id); Object.assign(p,fresh); renderCard(); if(fresh.status!=='PROCESSING'&&fresh.status!=='QUEUED'){ clearInterval(procTimer); procTimer=null; if(fresh.status==='TRANSCRIBED') toast(t('Video đã sẵn sàng để chỉnh sửa!','Your video is ready to edit!'),'success'); } }catch(e){} },CFG.pollMs); }

/* =========================================================
   9) CÀI ĐẶT / HẠN NGẠCH / GÓI & THANH TOÁN / HỖ TRỢ / HƯỚNG DẪN
   ========================================================= */
function pageSettings(root){ if(!S.settings){ root.append(h('div',{class:'card'},h('div',{style:{padding:'20px'}},t('Đang tải cài đặt…','Loading settings…')))); return; } const pv=S.settings.providers, keys=S.settings.keys;
  const setP=(k,v,payload)=>{ pv[k]=v; renderPage(); api.saveProviders(payload).catch(()=>toast(t('Lưu cài đặt thất bại','Failed to save settings'),'error')); };
  const providerRows=STT_PROVIDERS.map(pr=>[t('STT','STT'),pr,pv.stt,k=>setP('stt',k,{stt_provider:k})]).concat(TX_PROVIDERS.map(pr=>[t('Dịch','Translation'),pr,pv.translate,k=>setP('translate',k,{translate_provider:k})])).concat(TTS_PROVIDERS.map(pr=>[t('TTS','TTS'),pr,pv.tts,k=>setP('tts',k,{tts_provider:k})]));
  root.append(h('div',{class:'card'},h('div',{style:{padding:'18px 20px'}},h('div',{class:'panel-head'},t('CẤU HÌNH NHÀ CUNG CẤP','PROVIDER CONFIGURATION')),h('div',{class:'table-wrap'},h('table',{class:'table'},h('thead',null,h('tr',null,h('th',null,t('Nhóm','Group')),h('th',null,t('Nhà cung cấp','Provider')),h('th',null,t('Trạng thái','Status')),h('th',null,t('Thao tác','Action')))),h('tbody',null,...providerRows.map(([group,pr,cur,setter])=>h('tr',null,h('td',null,group),h('td',null,h('b',null,pr.name)),h('td',null,h('span',{class:'badge badge-'+(cur===pr.id?'success':'neutral')},cur===pr.id?t('Đang dùng','Active'):t('Sẵn sàng','Available'))),h('td',null,btn({v:cur===pr.id?'primary':'outline',label:cur===pr.id?t('Đã chọn','Selected'):t('Chọn','Select'),onClick:()=>setter(pr.id)}))))))))));
  const apiKeysCard=h('div',{class:'card'},h('div',{class:'panel-head'},t('API KEYS','API KEYS')),h('div',{style:{padding:'14px 20px',display:'flex',flexDirection:'column',gap:'10px'}}));
  const keysBody=apiKeysCard.lastChild;
  [['groq','Groq'],['gemini','Gemini'],['xkira','xKiro']].forEach(([id,name])=>{ const input=h('input',{class:'input',type:'password',placeholder:keys[id]?'••••••••••••':t('Chưa có key','No key saved')}); keysBody.append(h('div',{style:{display:'grid',gridTemplateColumns:'110px 1fr auto',gap:'8px',alignItems:'center'}},h('span',{style:{fontSize:'12.5px',fontWeight:600}},name),input,btn({v:'outline',label:t('Lưu','Save'),onClick:async()=>{ if(!input.value.trim()) return; try{ await api.saveApiKey(id,input.value.trim()); keys[id]=true; input.value=''; input.placeholder='••••••••••••'; toast(t(name+' đã lưu API key','API key saved for '+name),'success'); }catch(e){ toast(e.message,'error'); } }}))); });
  root.append(apiKeysCard); }
function pageQuota(root){ const q=S.quota||{used:0,total:30}; const pct=Math.min(100,Math.round((q.used/Math.max(1,q.total))*100));
  root.append(h('div',{class:'card'},h('div',{style:{padding:'22px'}},h('div',{style:{display:'flex',alignItems:'center',gap:'10px',marginBottom:'14px'}},h('div',{class:'qic',style:{width:40,height:40}},ic('gauge',18)),h('div',null,h('b',{style:{fontSize:'15px',display:'block'}},t('Hạn mức xử lý hàng tháng','Monthly processing quota')),h('span',{class:'muted',style:{fontSize:'12px'}},t('Đặt lại vào đầu mỗi tháng','Resets at the start of each month')))),h('div',{class:'progress'},h('i',{style:{width:pct+'%'}})),h('div',{style:{display:'flex',justifyContent:'space-between',marginTop:'8px'},class:'muted'},h('span',null,`${q.used} / ${q.total} ${t('phút','min')}`),h('span',null,pct+'%')))));
  root.append(h('div',{class:'card',style:{marginTop:'14px'}},h('div',{style:{padding:'18px 22px'}},h('p',{class:'hint',style:{margin:'0 0 12px',fontSize:'13px'}},t('Cần thêm phút xử lý? Nâng cấp gói hoặc dùng API key riêng để không tốn vào hạn mức hệ thống.','Need more processing minutes? Upgrade your plan or bring your own API keys to bypass the shared quota.')),btn({v:'primary',icon:'rocket',label:t('Xem các gói','View plans'),onClick:()=>go('billing')})))); }
function pageBilling(root){ const plans=[ [t('Miễn phí','Free'),'0',30,[t('30 phút / tháng','30 minutes / month'),t('Phụ đề & lồng tiếng cơ bản','Basic subtitles & dubbing')]], [t('Pro','Pro'),'19',300,[t('300 phút / tháng','300 minutes / month'),t('Chọn model dịch nâng cao','Advanced translation models'),t('Kết xuất 1080p','1080p rendering')]], [t('Max','Max'),'49',1500,[t('1.500 phút / tháng','1,500 minutes / month'),t('Phân tích giọng nói & bối cảnh','Voice & visual analysis'),t('Kết xuất 4K','4K rendering')]] ];
  root.append(h('div',{class:'plan-grid'},plans.map(([name,price,mins,feats],i)=>h('div',{class:'card plan-card'},h('b',{class:'nm'},name),h('div',{class:'price'},'$'+price,h('small',null,'/ '+t('tháng','mo'))),h('div',{class:'hint'},mins+' '+t('phút xử lý','processing minutes')),h('ul',null,feats.map(f=>h('li',{style:{display:'flex',alignItems:'center',gap:'6px'}},ic('check',13),f))),btn({v:i===1?'primary':'outline',label:i===0?t('Gói hiện tại','Current plan'):t('Nâng cấp','Upgrade'),disabled:i===0,cls:'btn-block',onClick:()=>toast(t('Kết nối cổng thanh toán để nâng cấp thật.','Connect a payment gateway to enable real upgrades.'),'info')}))))); }
function pageSupport(root){ root.append(h('div',{class:'card'},h('div',{style:{padding:'22px',display:'flex',flexDirection:'column',gap:'14px'}},h('div',{style:{display:'flex',alignItems:'center',gap:'10px'}},h('div',{class:'qic',style:{width:40,height:40}},ic('lifeBuoy',18)),h('b',{style:{fontSize:'15px'}},t('Cần trợ giúp?','Need help?'))),h('p',{class:'hint',style:{margin:0,fontSize:'13px'}},t('Gửi email cho đội ngũ Lingvo, chúng tôi thường phản hồi trong vòng 1 ngày làm việc.','Email the Lingvo team — we usually reply within one business day.')),h('div',{style:{display:'flex',gap:'8px'}},btn({v:'primary',icon:'mail',label:'support@lingvo.ai',onClick:()=>toast(t('Đã sao chép email hỗ trợ','Support email copied'),'success')}),btn({v:'outline',icon:'bookOpen',label:t('Xem hướng dẫn','View tutorial'),onClick:()=>go('tutorial')}))))); }
function pageTutorial(root){ const steps=[ [t('Nhập video','Import a video'),t('Dán link từ 12 nền tảng hoặc tải file lên trực tiếp.','Paste a link from 12 platforms or upload a file directly.')], [t('Chọn ngôn ngữ & nhà cung cấp','Choose languages & providers'),t('Chọn ngôn ngữ đích, bật lồng tiếng và chọn nhà cung cấp AI phù hợp.','Pick the target language, enable dubbing and choose the right AI providers.')], [t('Chờ phân tích','Wait for analysis'),t('Lingvo tách âm thanh, nhận dạng giọng nói và chuẩn bị phân đoạn.','Lingvo extracts audio, transcribes speech and prepares segments.')], [t('Chỉnh sửa & kết xuất','Edit & render'),t('Sửa bản dịch, tạo lồng tiếng, tùy chỉnh kiểu phụ đề rồi kết xuất video.','Edit translations, generate dubbing, style subtitles, then render the video.')] ];
  root.append(h('div',{class:'card'},h('div',{style:{padding:'20px'}},steps.map(([b,p],i)=>h('div',{style:{display:'flex',gap:'12px',padding:'10px 0',borderBottom:i<steps.length-1?'1px solid var(--border)':'none'}},h('div',{style:{width:26,height:26,borderRadius:'50%',background:'var(--primary)',color:'var(--primary-foreground)',display:'grid',placeItems:'center',fontSize:11,fontWeight:700,flexShrink:0}},i+1),h('div',null,h('b',{style:{fontSize:'13px',display:'block',marginBottom:'2px'}},b),h('span',{class:'hint'},p))))),h('div',{style:{padding:'0 20px 20px'}},btn({v:'primary',icon:'plus',label:t('Bắt đầu tạo dự án','Start a new project'),onClick:()=>go('new')})))); }

/* =========================================================
   10) ROUTER + KHỞI ĐỘNG
   ========================================================= */
function renderPage(){ const root=$('#contentInner'); root.innerHTML=''; if(!api){ root.append(emptyState('alert',t('Chưa kết nối dữ liệu','Not connected yet'),t('Kết nối backend hoặc xem dữ liệu minh hoạ ở thanh thông báo phía trên.','Connect a backend or preview sample data using the bar above.'),true)); return; }
  const map={ dashboard:pageDashboard, new:pageNewProject, processing:pageProcessing, videos:pageVideos, projects:pageProjects, outputs:pageOutputs, autopublish:pageAutopublish, settings:pageSettings, quota:pageQuota, billing:pageBilling, support:pageSupport, tutorial:pageTutorial };
  (map[S.route]||pageDashboard)(root); }
function renderAll(){ document.documentElement.lang=lang; renderConnBar(); renderSidebar(); renderTopbar(); renderPage(); }
$('#sbScrim').addEventListener('click',()=>{ S.mobileNav=false; $('#sbScrim').hidden=true; renderSidebar(); });
window.addEventListener('resize',()=>{ if(!isMobile()){ S.mobileNav=false; $('#sbScrim').hidden=true; } renderSidebar(); renderTopbar(); });
window.addEventListener('message',e=>{ const d=e.data; if(!d||typeof d!=='object') return; if(d.type==='lingvo:refresh'&&api) loadWorkspace(); });
async function init(){ renderConnBar(); renderSidebar(); renderTopbar(); renderPage(); await connectBackend(); if(api){ try{ S.quota=await api.getQuota(); renderPage(); }catch(e){} } }
init();
})();
