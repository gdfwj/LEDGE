'use strict';
let selectedRun=null, executionPoll=null, comparisons=[], observedRunId=null, observationVersion=0, displayedResultId=null, runStarting=false, importedRun=false;
const terminalStates=new Set(['complete','failed','cancelled']);
for(const button of document.querySelectorAll('[data-close]'))button.onclick=()=>$(button.dataset.close).close();
$('historyBtn').onclick=()=>{$('historyDialog').showModal();refreshHistory().catch(e=>status(e.message,true));};
$('runDetailsBtn').onclick=()=>$('runDetailsDialog').showModal();
function updateRunControls(){
 const active=selectedRun&&!importedRun&&!terminalStates.has(selectedRun.state);
 $('executionStart').textContent=runStarting?'Starting…':'Run simulation';
 $('executionStart').disabled=runStarting||!!active;
 $('executionCancel').disabled=!active;
 $('executionCancel').hidden=!active;
 $('runDetailsBtn').hidden=!$('executionDetail').textContent;
 $('compareRun').disabled=!result||!selectedRun?.result;
 $('exportResults').disabled=!result;
 for(const option of $('exportFormat').options)option.disabled=!result&&option.value!=='run';
 if(!result)$('exportFormat').value='run';
}
function detachRun(){
 clearTimeout(executionPoll);observationVersion++;selectedRun=null;observedRunId=null;displayedResultId=null;importedRun=false;
 $('executionStatus').textContent='';$('executionDetail').textContent='';updateRunControls();
}
function loadConfiguration(config,{detach=true,announce=true}={}){
 if(!config?.topology||!Array.isArray(config.groups)||!Array.isArray(config.requests))throw Error('Configuration needs topology, groups and requests.');
 if((config.execution_mode&&config.execution_mode!=='simulation')||config.task||(config.clock_policy&&config.clock_policy!=='offline'))throw Error('Only simulation configurations are supported.');
 if(detach)detachRun();
 exp=clone(config);selected.clear();selectedLink=null;
 $('loggingLevel').value=['off','minimal'].includes(exp.logging_level)?'off':'trace';
 changed(false);renderAll();
 if(announce)status('Loaded '+(config.name||'configuration'));
}
$('loggingLevel').onchange=()=>{exp.logging_level=$('loggingLevel').value;};
function executionConfig(){return {...clone(exp),logging_level:$('loggingLevel').value};}
$('validateBtn').onclick=async()=>{
 const button=$('validateBtn');button.disabled=true;button.textContent='Validating…';status('Validating…');
 try{const r=await api('/api/validate',{experiment:executionConfig()});status('Validation passed.');$('runDetailsTitle').textContent='Validation details';$('executionDetail').textContent=JSON.stringify(r,null,2);updateRunControls();}
 catch(e){status('Validation failed: '+e.message,true);}
 finally{button.disabled=false;button.textContent='Validate configuration';}
};
async function startRun(config){
 runStarting=true;updateRunControls();status('Starting…');
 try{
  const r=await api('/api/runs',{experiment:config});
  detachRun();selectedRun=r;localStorage.setItem('ledge-simulation-active-run',r.id);
  await refreshHistory();await observeRun(r.id);return true;
 }catch(e){status('Run failed: '+e.message,true);return false;}
 finally{runStarting=false;updateRunControls();}
}
$('executionStart').onclick=()=>{try{startRun(executionConfig());}catch(e){status(e.message,true);}};
$('executionCancel').onclick=async()=>{
 if(!selectedRun||importedRun||terminalStates.has(selectedRun.state))return;
 $('executionCancel').disabled=true;
 try{await api('/api/cancel/'+selectedRun.id,{});$('executionStatus').textContent='Cancelling…';await observeRun(selectedRun.id);}
 catch(e){status('Cancel failed: '+e.message,true);updateRunControls();}
};
async function refreshHistory(){
 const rows=await api('/api/runs');
 const current=selectedRun?.id||localStorage.getItem('ledge-simulation-active-run');
 const exists=rows.some(r=>r.id===current);
 if(current&&!exists&&!importedRun)localStorage.removeItem('ledge-simulation-active-run');
 $('runHistory').innerHTML=options([['',rows.length?'Select a run':'No saved runs'],...rows.map(r=>[r.id,`${r.config_name||r.id} · ${r.state}`])],exists?current:'');
 return rows;
}
function displayRunResult(record){
 const r=record.result;
 if(!r?.input?.topology||!Array.isArray(r.requests)||!r.summary)return false;
 exp=clone(r.input);selected.clear();selectedLink=null;renderAll();acceptResult(r);
 return true;
}
function showRunDetails(r){
 $('runDetailsTitle').textContent='Run details';
 $('executionDetail').textContent=JSON.stringify({id:r.id,state:r.state,elapsed_s:r.elapsed_s,engine_events:r.result?.engine_event_count,error:r.error},null,2);
}
async function observeRun(id){
 if(!id)return;
 const version=++observationVersion;clearTimeout(executionPoll);$('executionCancel').disabled=true;
 const r=await api('/api/runs/'+id);if(version!==observationVersion)return;
 const newSelection=observedRunId!==id;selectedRun=r;observedRunId=id;importedRun=false;
 localStorage.setItem('ledge-simulation-active-run',id);
 if(newSelection){
  displayedResultId=null;
  if(r.config.topology)loadConfiguration(r.config,{detach:false,announce:false});else changed(false);
 }
 $('executionStatus').textContent=`Elapsed ${fmt(r.elapsed_s)} s`;
 showRunDetails(r);
 if(r.result&&displayedResultId!==id){
  displayedResultId=id;
  if(!displayRunResult(r))status('Run complete.');
 }
 if(r.state==='failed')status('Run failed: '+(r.error||'See run details.'),true);
 else if(r.state==='cancelled')status('Run cancelled.');
 else if(!terminalStates.has(r.state))status('Running…');
 updateRunControls();
 if(terminalStates.has(r.state))await refreshHistory();
 else executionPoll=setTimeout(()=>observeRun(id).catch(e=>status('Connection lost: '+e.message,true)),1000);
}
$('refreshRuns').onclick=async()=>{try{await refreshHistory();if(selectedRun&&!importedRun)await observeRun(selectedRun.id);}catch(e){status(e.message,true);}};
$('runHistory').onchange=()=>{if($('runHistory').value)observeRun($('runHistory').value).catch(e=>status(e.message,true));};
async function reconnectHistory(){
 const rows=await refreshHistory(),id=localStorage.getItem('ledge-simulation-active-run');
 if(id&&rows.some(r=>r.id===id))await observeRun(id);
}
function importSavedResults(text){
 const value=JSON.parse(text),record=value.result?value:null,r=record?record.result:value;
 if(!r?.input?.topology||!Array.isArray(r.requests)||!r.summary)throw Error('Import a full run or a saved simulation result.');
 const config=record?.config||r.input;
 loadConfiguration(config,{announce:false});
 importedRun=true;selectedRun=record||{id:'imported',state:'complete',config:clone(config),result:r};
 displayedResultId=selectedRun.id;displayRunResult(selectedRun);showRunDetails(selectedRun);
 $('executionStatus').textContent=record?.elapsed_s!=null?'Elapsed '+fmt(record.elapsed_s)+' s':'';updateRunControls();status('Results imported.');
}
$('importResults').onclick=()=>pickFile('.json',importSavedResults);
$('pasteResults').onclick=()=>$('logDialog').showModal();
$('cancelLogImport').onclick=()=>$('logDialog').close();
$('applyLogImport').onclick=()=>{try{importSavedResults($('logText').value);$('logDialog').close();}catch(e){status(e.message,true);}};
$('exportResults').onclick=()=>{
 const format=$('exportFormat').value;
 if(format==='run'){
  const value=selectedRun?.result?selectedRun:result;
  if(value)download((selectedRun?.id||'run')+'.json',value);
 }else if(!result)status('Select a run with request results.',true);
 else if(format==='results')download('metrics.json',{requests:result.requests,summary:result.summary});
 else if(format==='csv')download('requests.csv',csv(result.requests,['request_id','instance','model','arrival_s','start_s','first_token_s','completion_s','queue_s','service_s','response_s','ttft_s','tpot_s']),'text/csv');
 else download('resolved-configuration.json',{input:result.input,resolved:result.resolved,summary:result.summary});
};
function comparisonRows(record){return record.result?.requests?.map(x=>({...x,id:x.request_id}))||[];}
function renderComparisons(){
 if(!comparisons.length){$('runComparison').innerHTML='';return;}
 $('runComparison').innerHTML='<div class="tablewrap"><table><thead><tr><th>Run / request</th><th>Elapsed / s</th><th>TTFT / s</th><th>TPOT / s</th><th>Service / s</th><th></th></tr></thead><tbody>'+comparisons.flatMap(r=>comparisonRows(r).map(x=>{
  const completion=x.service_s??(x.output_tokens===1?x.ttft_s:Number.isFinite(x.output_tokens)&&x.tpot_s!=null?x.ttft_s+(x.output_tokens-1)*x.tpot_s:null);
  return `<tr><td>${esc(r.id)} / ${esc(x.id)}</td><td>${fmt(r.elapsed_s)}</td><td>${fmt(x.ttft_s)}</td><td>${fmt(x.tpot_s)}</td><td>${fmt(completion)}</td><td><button data-remove-comparison="${esc(r.id)}">Remove</button></td></tr>`;
 })).join('')+'</tbody></table></div>';
 $('runComparison').querySelectorAll('[data-remove-comparison]').forEach(b=>b.onclick=()=>{comparisons=comparisons.filter(r=>r.id!==b.dataset.removeComparison);renderComparisons();});
}
$('compareRun').onclick=()=>{if(!selectedRun?.result)return;comparisons=comparisons.filter(r=>r.id!==selectedRun.id);comparisons.push(clone(selectedRun));renderComparisons();};
initialize().catch(e=>status(e.message,true));
