clear all
close all
clc


% geração dos dados
% A = gendatb([500 500]); [C,D] = gendat(A,0.2);
% xn = A.data;
% Maximo = max(xn);
% Minimo = min(xn);
% for i = 1:length(xn)
%     x(i,:) = (xn(i,:) - Minimo)./ (Maximo-Minimo) ;
% end
% 
% y = [repmat([0 1],500,1) ;repmat([1 0],500,1)]';

load('DadosExemplo.mat')
 %Parâmetros
Dn=x';

[LinD ColD]=size(x');

I=randperm(ColD);
Dn=Dn(:,I);
alvos=y(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

% Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=0.8;    % Porcentagem usada para treino
ptst=1-ptrn; % Porcentagem usada para teste

J=floor(ptrn*ColD);

% Vetores para treinamento e saidas desejadas correspondentes
P = Dn(:,1:J); T1 = alvos(:,1:J); 
[lP cP]=size(P);   % Tamanho da matriz de vetores de treinamento

% Vetores para teste e saidas desejadas correspondentes
Q = Dn(:,J+1:end); T2 = alvos(:,J+1:end); 
[lQ, cQ]=size(Q);   % Tamanho da matriz de vetores de teste


% DEFINE ARQUITETURA DA REDE
%===========================
Ne = 500; % No. de epocas de treinamento
Nh = 12;   % No. de nuronios na camada oculta
No = 2;   % No. de neuronios na camada de saida

alfa=0.3;   % Passo de aprendizagem

% Inicia matrizes de pesos
WW=0.2*rand(Nh,lP+1);   % Pesos entrada -> camada oculta

MM=0.2*rand(No,Nh+1);   % Pesos camada oculta -> camada de saida

%%% ETAPA DE TREINAMENTO
for t=1:Ne,
    
    Epoca=t,
    
    I=randperm(cP); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento e saidas desejadas
    
    EQ=0;
    for tt=1:cP,   % Inicia LOOP de epocas de treinamento
        % CAMADA OCULTA
        X=[-1; P(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)

        % CAMADA DE SAIDA 
        Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)

        % CALCULO DO ERRO 
        Ek = T1(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede
        
        %%% CALCULO DOS GRADIENTES LOCAIS
        Dk = Ok.*(1 - Ok);  % derivada da sigmoide logistica (camada de saida)
        DDk = Ek.*Dk;       % gradiente local (camada de saida)
        
        Di = Yi.*(1 - Yi); % derivada da sigmoide logistica (camada oculta)
        DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)

        % AJUSTE DOS PESOS - CAMADA DE SAIDA
        MM = MM + alfa*DDk*Y';
        
        % AJUSTE DOS PESOS - CAMADA OCULTA
        WW = WW + alfa*DDi*X';
        erroExemplo(tt,:) = Ek;
    end   % Fim de uma epoca
    
    erro(t) = sum(sum(erroExemplo.^2));
    % MEDIA DO ERRO QUADRATICO P/ EPOCA
end   % Fim do loop de treinamento


%% ETAPA DE TESTE  %%%
EQ2=0;
HID2=[];
OUT2=[];
for tt=1:cQ,
    % CAMADA OCULTA
    X=[-1; Q(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
    Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
    Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)
    
    % CAMADA DE SAIDA 
    Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
    Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
    Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
    OUT2=[OUT2 Ok];       % Armazena saida da rede
    
    
end


% CALCULA TAXA DE ACERTO
count_OK=0;  % Contador de acertos
for t=1:cQ,
    [T2max iT2max]=max(T2(:,t));  % Indice da saida desejada de maior valor
    [OUT2_max iOUT2_max]=max(OUT2(:,t)); % Indice do neuronio cuja saida eh a maior
    if iT2max==iOUT2_max,   % Conta acerto se os dois indices coincidem 
        count_OK=count_OK+1;
    end
end

% Taxa de acerto global
Tx_OK=100*(count_OK/cQ)


% Gera a figura da superfície de decisão
xc = 0:0.01:1;
yc = 0:0.01:1;

OUT3 = [];
IdxX = [];
IdxY = [];
for i = 1:length(xc)
    for j = 1:length(yc)
        X=[-1; xc(i) ;yc(j)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)
        
        % CAMADA DE SAIDA 
        Y=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
        OUT3=[OUT3 Ok];       % Armazena saida da rede
        
        [OUT3_max iOUT3_max]=max(Ok); % Indice do neuronio cuja saida eh a maior

        Result(i,j) = iOUT3_max;
        ResultIdxX(i,j) = xc(i);
        ResultIdxY(i,j) = yc(j);
        IdxX = [IdxX xc(i)];
        IdxY = [IdxY yc(j)];
    end
end
hold on
mesh(ResultIdxX,ResultIdxY,Result)
plot(x(1:500,1),x(1:500,2),'x')
hold on
plot(x(501:1000,1),x(501:1000,2),'rx')