function val = QMITL_Eval2raw(Sys,phi, traj)        
% QMITL_EVAL computes the satisfaction function of a property for one
% trace, variable time steps version   
%
% Usage: val = QMITL_Eval2(Sys, phi, traj, t)  
%
% 

%  if (numel(t)==1)      
%    ind_t = find(traj.time>t,1);
%    interval = [t traj.time(ind_t)];
%  else
%    interval = [t(1) t(end)];
%  end
  interval = [traj.time(1) traj.time(end)];
  
  [valarray time_values] = GetValues(Sys,phi,traj,interval);
  
  if isscalar(valarray)
    valarray= valarray*ones(1, numel(time_values));
  end

  %val = interp1(time_values, valarray, t,'linear',valarray(end));
  val.time = time_values;
  val.X = valarray;
    
function [valarray time_values] = GetValues(Sys,phi,traj,interval)
  
  for i=1:Sys.DimP
    eval([Sys.ParamList{i} '= traj.param(' num2str(i) ');']);       
  end
  
  switch (phi.type)
    
   case 'predicate'
    time_values= GetTimeValues(traj,interval);
    params = phi.params;
    params.Sys = Sys;
    evalfn = @(t) phi.evalfn(0,traj,t,params);
    
    try  % works if predicate can handle multiple time values
        valarray = evalfn(time_values);
    catch
        valarray = arrayfun(evalfn, time_values);
    end
    
   case 'not'
    [valarray time_values] = GetValues(Sys,phi.phi,traj,interval);
    valarray = - valarray;
   
   case 'or'
    [valarray1 time_values1] = GetValues(Sys,phi.phi1,traj,interval);
    [valarray2 time_values2] = GetValues(Sys,phi.phi2,traj,interval);
    [time_values valarray] = RobustOr(time_values1, valarray1, time_values2, valarray2);
       
   case 'and'
    [valarray1 time_values1] = GetValues(Sys,phi.phi1,traj,interval);
    [valarray2 time_values2] = GetValues(Sys,phi.phi2,traj,interval);
    [time_values valarray] = RobustAnd(time_values1, valarray1, time_values2, valarray2);
   
   case '=>'
    [valarray1 time_values1] = GetValues(Sys,phi.phi1,traj,interval);
    [valarray2 time_values2] = GetValues(Sys,phi.phi2,traj,interval);
    valarray1 = -valarray1;
    [time_values valarray] = RobustOr(time_values1, valarray1, time_values2, valarray2);
        
   case 'always'    
    DDDI = eval(phi.interval);
    next_interval = DDDI+interval;
    [valarray1 time_values1] = GetValues(Sys, phi.phi, traj, next_interval);   
    [time_values valarray] = RobustEv(time_values1, -valarray1, DDDI);
    valarray = -valarray;
    
   case 'eventually'
    DDDI = eval(phi.interval);
    next_interval =  DDDI+interval;
    [valarray1 time_values1] = GetValues(Sys,phi.phi, traj, next_interval);
    [time_values valarray] = RobustEv(time_values1, valarray1, DDDI);
    
   case 'until'  
    DDDI = eval(phi.interval);    
    interval1  =  [interval(1), DDDI(2)+interval(2)];
    interval2 =  DDDI+interval;
    
    [valarray1 time_values1]= GetValues(Sys,phi.phi1, traj, interval1);
    [valarray2 time_values2]= GetValues(Sys,phi.phi2, traj, interval2);
    [time_values valarray] = RobustUntil(time_values1, valarray1, time_values2, valarray2, DDDI);
    
    
  end
  
 
function time_values = GetTimeValues(traj,interval)
% TO BE CHECKED
  
  ind_ti = find(traj.time>= interval(1),1);
  ind_tf = find(traj.time> interval(end),1);
  
  if (ind_ti==1)
    ind_ti = 2;
  end
  
  if isempty(ind_tf)
    time_values = traj.time(ind_ti-1:end);
  else
    time_values = traj.time(ind_ti-1:ind_tf-1);
  end
  
  if (interval(1)<traj.time(1))
    time_values = [interval(1) time_values];
  end

  
  
function [v1 v2] = SyncValues(v1, v2)
  l1 = numel(v1);
  l2 = numel(v2);
  if l1==l2 
    return;
  end
  if (l1>l2)
   % v2 = [v2 zeros(1,l1-l2)];
   v1 = v1(1:l2);
  else
 %   v1 = [v1 zeros(1,l2-l1)];
   v2 = v2(1:l1);
  end
  