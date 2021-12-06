function [S,V,Q] = main(X_part,label,options)

% Naiyao Liang, Zuyuan Yang, Lingjiang Li, Zhenni Li, and Shengli Xie. 
% Incomplete Multiview Clustering with Cross-view Feature Transformation.
% IEEE Transactions on Artificial Intelligence (TAI), 2021.

options.c=numel(unique(label));
c=options.c;
v= size(X_part,2); 
lambda=options.lambda;
sigma=options.sigma;
gamma=options.gamma;
NITER = 30;
if nargin < 4
    k = 9;
end

options.X_part=X_part;
for p=1:length(X_part)
    X_0{p}=X_part{p}*options.W1{p}'; % the unobserved samples are filled with zero value
end
options.X_0=X_0;
%% Construct F, N, and B 
 [F,N,B]=algorithm234(options);
 options.B=B;
 options.N=N; 
 options.F=F; 
%% Initialization
[Q, Qi]=Q_ini(X_part,options);   %  Initialize Q
distX_initial=Construct_X(Q,options);
num=size(options.X_0{1},2); 
SUM = zeros(num);
for i = 1:v
    SUM = SUM + distX_initial(:,:,i);
end
distX = 1/v*SUM;
[distXs, idx] = sort(distX,2);
S = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distXs(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);    %initialize S
end
alpha = mean(rr);  

% initialize V
S = (S+S')/2;                      
D = diag(sum(S));
L = D - S;   
[V, temp, evs]=eig1(L, c, 0);

if sum(evs(1:c+1)) < 0.00000000001
    error('The original graph has more than %d connected component', c);
end
for i = 1 : v
    Wv{i} = 0.5./sqrt(sum(sum( distX_initial(:,:,i).*S)+eps));
end    

%% Alternating Optimization
for iter = 1:NITER
    for p=1:v
        for w=N{p}
            T_tu_1Q_other=0;
            for s=1:v
                if ~(s==p || s==w) 
                   T_tu_1Q_other=T_tu_1Q_other+Q{s,p}*X_0{s}*B{p}*L*B{p}'*X_0{w}'; 
                end    
            end    
            T_yinshe_1Q=X_0{p}*F{w,p}*F{w,p}'*X_0{w}'; 
            T_yinshe_2Q=X_0{w}*F{w,p}*F{w,p}'*X_0{w}';
            T_tu_1Q=2*Wv{p}*X_0{p}*L*B{p}'*X_0{w}';
            T_tu_2Q=2*Wv{p}*X_0{w}*B{p}*L*B{p}'*X_0{w}';
            Q{w,p}=(gamma*T_yinshe_1Q - T_tu_1Q - 2*Wv{p}*T_tu_1Q_other)/...
                    (gamma*T_yinshe_2Q+sigma*Qi{w,p}+T_tu_2Q);   
                Qi{w,p}=diag(0.5./sqrt(sum(Q{w,p}.*Q{w,p},1)));       
        end    
    end
    distX_updated_no_wei=Construct_X(Q,options);  
    SUM = zeros(num,num); SUM_no_wei=SUM;
    for i = 1 : v
        Wv{i} = 0.5./sqrt(sum(sum( distX_updated_no_wei(:,:,i).*S)+eps));
        SUM_no_wei = SUM_no_wei + distX_updated_no_wei(:,:,i);
        distX_updated_wei(:,:,i) = Wv{i}*distX_updated_no_wei(:,:,i); 
        SUM = SUM + distX_updated_wei(:,:,i);
    end
    distX = SUM;  
    distX_no_wei = 1/v*SUM_no_wei; 
    [~, idx] = sort(distX_no_wei,2); 
    
    %update S
    distf = L2_distance_1(V',V');
    S = zeros(num);
    for i=1:num        
        idxa0 = idx(i,2:k+1);  
        dfi = distf(i,idxa0);   
        dxi = distX(i,idxa0);   
        ad = -(dxi+lambda*dfi)/(2*alpha);
        S(i,idxa0) = EProjSimplex_new(ad);
    end
    
    %update F
    S = (S+S')/2;            
    D = diag(sum(S));
    L = D-S;
    V_old = V;
    [V, temp, ev]=eig1(L, c, 0);
    evs(:,iter+1) = ev;
    thre = 1*10^-10;
    fn1 = sum(ev(1:c));                                   
    fn2 = sum(ev(1:c+1));
    if fn1 > thre
        lambda = 2*lambda;
    elseif fn2 < thre
        lambda = lambda/2;  V = V_old;
    else
        break;
    end
    sprintf('iter = %d',iter)
end


end

function [Q, Qi]=Q_ini(X_part,options)
X=X_part; 
W1=options.W1; 
F=options.F;  
nv=length(X);
I=cell(1,nv);
Q=cell(nv,nv); Qi=Q;
for p=1:nv
    I{p}=eye(size(X{p},1)); 
end    
for p=1:nv
    for w=1:nv
        if p==w
            continue;
        end    
        Q{w,p}=(X{p}*W1{p}'*F{w,p}*F{w,p}'*W1{w}*X{w}')/...
                (X{w}*W1{w}'*F{w,p}*F{w,p}'*W1{w}*X{w}'+I{w});
        Q{w,p}=rand(size(Q{w,p}));    
        Qi{w,p}=diag(0.5./sqrt(sum(Q{w,p}.*Q{w,p},1)));    
    end    
end    
end

function [F,N,B1]=algorithm234(options)
W1=options.W1;
W0=options.W0;
nv=length(W1);
numOfDatas=size(W1{1},1);
F=cell(nv,nv); G=F;   
for iv = 1:nv  
    for jv=1:nv 
        if iv==jv
            continue;
        end
        F{iv,jv}=(W1{iv}*W1{iv}').*(W1{jv}*W1{jv}'); 
        G{iv,jv}=W1{iv}'*W0{jv};
    end
end
B=cell(1,nv); 
for p=1:nv
     B{p}=0;
    for w=1:nv
        if w==p
            continue;
        end    
        B{p}=B{p}+G{w,p}'*G{w,p};
    end    
    B{p}=diag(1./diag(B{p}));
end    
    B1=cell(1,nv); 
for p=1:nv
     B1{p}=W0{p}*B{p}*W0{p}';  
end     
N=cell(nv,1);
for p=1:nv
    N{p}=[]; 
    for z=1:nv
        if z==p || isequal(F(z,p),zeros(numOfDatas,numOfDatas))
            continue;
        end
        N{p}=[N{p} z];
    end    
end    
end
   
function [distX_initial,Xa]=Construct_X(Q,options)
nv=length(options.X_0); 
X_0=options.X_0;
B=options.B;
N=options.N;
  H=cell(1,nv);  
  for p=1:nv
      H{p}=0;
      for w=N{p}
         H{p}=H{p}+Q{w,p}*X_0{w}; 
      end  
      Xa{p}=X_0{p}+H{p}*B{p};
  end    
for p=1:nv
    distX_initial(:,:,p) =  L2_distance_1( Xa{p},Xa{p} ) ; 
end
end