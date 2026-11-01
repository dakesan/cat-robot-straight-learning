#pragma once

const char CONTROL_PAGE[] PROGMEM = R"HTML(<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>猫じゃらしロボ</title><style>
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:#f7f7f3;color:#202622;margin:0;padding:20px;line-height:1.6}main{max-width:620px;margin:auto}h1{font-size:25px;margin:0 0 8px}section{background:white;border:1px solid #d4dad5;border-radius:14px;padding:18px;margin:14px 0}h2{font-size:18px;margin:0 0 12px}button{font:inherit;min-height:52px;border:1px solid #829489;border-radius:9px;background:#edf4ee;color:#14251b;padding:9px 16px;touch-action:none;cursor:pointer}button:disabled{opacity:.45;cursor:default}.stop{background:#a82727;color:white;border-color:#a82727;font-weight:700}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.row>*{flex:1}.drive{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}.drive button{height:65px}.drive .forward{grid-column:2}.drive .left{grid-column:1}.drive .back{grid-column:2}.status{font-size:15px;overflow-wrap:anywhere}label{display:block;margin:10px 0}input[type=range]{width:100%;height:30px}input[type=checkbox]{width:20px;height:20px;vertical-align:middle}small{color:#526359}output{font-variant-numeric:tabular-nums}#notice{color:#862020;min-height:26px}details{margin-top:10px}summary{cursor:pointer}code{font-size:13px}p{margin:7px 0}
button,button *{-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
</style></head><body><main>
<h1>猫じゃらしロボ</h1><p>起動時は停止。走行ボタンを離すと停止します。</p>
<section><div class="row"><button id="arm">操作を有効にする</button><button id="stop" class="stop">すべて停止</button></div>
<p id="notice" role="status">状態を取得しています</p><p id="state" class="status"></p></section>
<section><h2>走行</h2><label>モータ出力 <output id="speedValue">20</output>%<input id="speed" type="range" min="15" max="50" value="20" step="1"></label>
<div class="drive"><button class="forward" data-left="1" data-right="1">前へ</button><button class="left" data-left="1" data-right="-1">左旋回</button><button id="driveStop" class="stop">走行停止</button><button data-left="-1" data-right="1">右旋回</button><button class="back" data-left="-1" data-right="-1">後ろへ</button></div>
<small>ボタンを押している間だけ走行。表示はPWM出力の割合で、速度の実測値ではありません。</small></section>
<section><h2>じゃらし</h2><label><input id="servoEnabled" type="checkbox"> サーボを動かす</label>
<label>左 <input id="servo" type="range" min="-100" max="100" value="0"> 右</label><small>初期の振り幅は中心付近です。停止するとサーボ信号も止めます。</small></section>
<section><h2>回転の読み取り</h2><p id="counts">左 0 ／ 右 0</p><p id="battery">電圧監視：無効</p><small>encoderは回転を数えるセンサーです。表示はパルスのカウントで、距離やrpmへの換算前の値です。</small>
<details><summary>停止中の設定</summary><button id="reverseHeading">前後を入れ替える</button><label><input id="invertLeft" type="checkbox"> 左モータの回転方向を反転</label><label><input id="invertRight" type="checkbox"> 右モータの回転方向を反転</label>
<label><input id="batteryEnabled" type="checkbox"> 電池の分圧抵抗30 kΩ／10 kΩを接続済み（電圧監視を有効化）</label><button id="save">設定を保存</button>
<p><small>電圧監視は分圧回路の接続確認後に有効化します。公称抵抗で6.8 V未満が200 ms続くとソフト停止します。電池の自動遮断ではありません。</small></p></details></section>
<p><small>通信が300 ms途絶えると操作を無効にします。モータ出力中にencoderの変化が800 ms途絶えた場合も停止します。再開には「操作を有効にする」が必要です。猫を認識して自動で動く機能は含みません。</small></p>
</main><script>
'use strict';
const token='__TOKEN__';
const byId=id=>document.getElementById(id);
const reasons={startup:'起動後・停止中',operator:'操作による停止',command_timeout:'通信が途絶えたため停止',encoder_left_no_response:'左encoderの応答がないため停止',encoder_right_no_response:'右encoderの応答がないため停止',battery_low:'電池電圧低下で停止',battery_sensor:'電池電圧を確認できないため停止',invalid_command:'指令の値が範囲外のため停止',output_error:'PWM初期化または出力の異常',network_error:'Wi-Fi初期化の異常'};
let armed=false,epoch=0,sequence=0,dir=[0,0],sending=false,generation=0;
let lastStatus=null,activePointer=null,polling=false,configDirty=false;
async function request(path,body){
 const response=await fetch(path,{method:body?'POST':'GET',headers:body?{'X-Robot-Token':token,'Content-Type':'application/x-www-form-urlencoded'}:{},body:body?new URLSearchParams(body):undefined,cache:'no-store',signal:AbortSignal.timeout(1200)});
 const value=await response.json();if(!response.ok)throw new Error(value.error||'通信エラー');return value;
}
function setArmed(value){armed=value;byId('arm').disabled=value;for(const button of document.querySelectorAll('[data-left]'))button.disabled=!value;byId('servoEnabled').disabled=!value;}
function showStatus(status){lastStatus=status;byId('state').textContent=`${status.firmware} ／ 左出力 ${status.left}% ／ 右出力 ${status.right}%`;
 byId('counts').textContent=`左 ${status.encoder_left} ／ 右 ${status.encoder_right}`;
 byId('battery').textContent=status.battery_enabled?`電池電圧 ${(status.battery_mv/1000).toFixed(2)} V`:'電圧監視：無効（分圧回路の接続未確認）';
 if(!status.armed){if(armed){generation++;dir=[0,0];byId('servoEnabled').checked=false;}setArmed(false);byId('notice').textContent=reasons[status.reason]||status.reason;}
 if(!armed&&!configDirty){byId('invertLeft').checked=status.invert_left;byId('invertRight').checked=status.invert_right;byId('batteryEnabled').checked=status.battery_enabled;}
}
async function poll(){if(polling)return;polling=true;const current=generation;try{const state=await request('/api/status');if(current===generation)showStatus(state);}catch{if(armed)stopAll();byId('notice').textContent='接続を確認してください。通信停止時は自動で停止します。';}finally{polling=false;}}
async function sendCommand(){if(!armed||sending)return;sending=true;const current=generation;const speed=Number(byId('speed').value);
 try{await request('/api/command',{epoch,seq:++sequence,left:dir[0]*speed,right:dir[1]*speed,servo:byId('servoEnabled').checked?1500+Number(byId('servo').value):0});}
 catch(error){if(current===generation){stopAll();byId('notice').textContent=error.message;}}finally{sending=false;}
}
function stopAll(){generation++;dir=[0,0];activePointer=null;byId('servoEnabled').checked=false;setArmed(false);fetch('/api/stop',{method:'POST',headers:{'X-Robot-Token':token},keepalive:true}).catch(()=>{});}
byId('arm').onclick=async()=>{const current=++generation;byId('arm').disabled=true;try{const state=await request('/api/arm',{});if(current!==generation){stopAll();return;}epoch=state.epoch;sequence=0;dir=[0,0];byId('servoEnabled').checked=false;setArmed(true);byId('notice').textContent='操作可能・走行ボタンを押すと動きます';await sendCommand();}catch(error){byId('notice').textContent=error.message;}finally{if(!armed)byId('arm').disabled=false;}};
byId('stop').onclick=stopAll;
byId('driveStop').onclick=()=>{dir=[0,0];sendCommand();};
byId('speed').oninput=()=>{byId('speedValue').value=byId('speed').value;};
for(const button of document.querySelectorAll('button')){
 for(const eventName of ['selectstart','contextmenu','dragstart'])button.addEventListener(eventName,event=>event.preventDefault());
}
for(const button of document.querySelectorAll('[data-left]')){
 button.addEventListener('touchstart',event=>event.preventDefault(),{passive:false});
 button.onpointerdown=event=>{if(event.button!==0||!armed||activePointer!==null)return;event.preventDefault();activePointer=event.pointerId;button.setPointerCapture(event.pointerId);dir=[Number(button.dataset.left),Number(button.dataset.right)];sendCommand();};
 const release=event=>{if(event.pointerId!==activePointer)return;activePointer=null;dir=[0,0];sendCommand();};button.onpointerup=release;button.onpointercancel=release;button.onlostpointercapture=release;
}
for(const id of ['invertLeft','invertRight','batteryEnabled'])byId(id).onchange=()=>{configDirty=true;};
byId('save').onclick=async()=>{try{if(armed)throw new Error('すべて停止してから保存してください');const state=await request('/api/config',{invert_left:byId('invertLeft').checked?1:0,invert_right:byId('invertRight').checked?1:0,battery_enabled:byId('batteryEnabled').checked?1:0});configDirty=false;showStatus(state);}catch(error){byId('notice').textContent=error.message;}};
byId('reverseHeading').onclick=()=>{if(armed){byId('notice').textContent='すべて停止してから変更してください';return;}byId('invertLeft').checked=!byId('invertLeft').checked;byId('invertRight').checked=!byId('invertRight').checked;configDirty=true;byId('save').click();};
window.addEventListener('pagehide',stopAll);document.addEventListener('visibilitychange',()=>{if(document.hidden)stopAll();});
setArmed(false);poll();setInterval(poll,500);setInterval(sendCommand,100);
</script></body></html>)HTML";
