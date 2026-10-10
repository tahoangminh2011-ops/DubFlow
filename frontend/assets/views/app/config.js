/* =========================================================
   CẤU HÌNH BACKEND — sửa tại đây hoặc truyền ?api=&token=&plan= trên URL
   apiBase rỗng => chạy chế độ demo (dữ liệu giả lập, không cần backend)
   ========================================================= */
window.LINGVO_CONFIG = { apiBase: window.location.origin, token: "", plan: "MAX", pollMs: 1500 };
(function(){try{if(localStorage.getItem("atv_dark")==="false")document.documentElement.classList.remove("dark");}catch(e){}})();
