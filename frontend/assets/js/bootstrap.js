(async function(){
  const root=document.getElementById('root');
  const q=new URLSearchParams(location.search);
  const page=q.get('page')==='editor'?'editor':'app';
  const brandName=(window.LINGVO_BRANDING&&window.LINGVO_BRANDING.name)||'Lingvo';
  document.title=page==='editor'?brandName+' · Editor':brandName+' · Không gian làm việc';
  const base='/frontend/assets/views/'+page+'/';
  const get=async(name)=>{const r=await fetch(base+name,{cache:'no-cache'});if(!r.ok)throw new Error('Không tải được giao diện: '+name);return await r.text();};
  try{
    const [config,css,body]=await Promise.all([get('config.js'),get('style.css'),get('body.html')]);
    const cfg=document.createElement('script');cfg.textContent=config;document.head.appendChild(cfg);
    const style=document.createElement('style');style.id='lingvo-view-style';style.textContent=css;document.head.appendChild(style);
    root.innerHTML=body;
    const code=await get('app.js');const app=document.createElement('script');app.textContent=code;document.body.appendChild(app);
  }catch(e){root.textContent=(e&&e.message)||'Không thể tải giao diện Lingvo.';console.error(e);}
})();
