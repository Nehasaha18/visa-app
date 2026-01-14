document.addEventListener('DOMContentLoaded',function(){
  // Smooth scrolling for internal links
  document.querySelectorAll('a[href^="#"]').forEach(a=>{
    a.addEventListener('click',e=>{
      const href=a.getAttribute('href');
      if(href.length>1){
        e.preventDefault();
        document.querySelector(href).scrollIntoView({behavior:'smooth',block:'start'});
      }
    });
  });

  // Intersection observer for fade-in
  const io=new IntersectionObserver((entries)=>{
    entries.forEach(ent=>{
      if(ent.isIntersecting) ent.target.classList.add('visible');
    });
  },{threshold:0.12});
  document.querySelectorAll('.fade-in, .card, .chart-card, .panel-card').forEach(el=>io.observe(el));

  // Charts (sample data)
  try{
    const ctx=document.getElementById('chartTrend');
    if(ctx){
      new Chart(ctx,{type:'line',data:{labels:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'],datasets:[{label:'Processing Days',data:[45,42,40,38,36,37,35,34,38],borderColor:'#7b61ff',backgroundColor:'rgba(123,97,255,0.12)',fill:true,tension:0.35,borderWidth:3}]},options:{plugins:{legend:{labels:{color:'white',font:{size:14,weight:'600'}}}},scales:{y:{ticks:{color:'white',font:{size:13,weight:'600'}},grid:{color:'rgba(255,255,255,0.05)'},beginAtZero:false},x:{ticks:{color:'white',font:{size:13,weight:'600'}},grid:{color:'rgba(255,255,255,0.05)'}}}}});
    }

    const ctx2=document.getElementById('chartSeason');
    if(ctx2){
      new Chart(ctx2,{type:'bar',data:{labels:['Peak','Off-Peak'],datasets:[{label:'Avg Days',data:[48,36],backgroundColor:['#0a61ff','#7b61ff']}]},options:{plugins:{legend:{labels:{color:'white',font:{size:14,weight:'600'}}}},scales:{y:{ticks:{color:'white',font:{size:13,weight:'600'}},grid:{color:'rgba(255,255,255,0.05)'}},x:{ticks:{color:'white',font:{size:13,weight:'600'}},grid:{color:'rgba(255,255,255,0.05)'}}}}});
    }
  }catch(e){console.warn(e)}
});
