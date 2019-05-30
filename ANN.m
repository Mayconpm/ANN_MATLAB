function [NET, OUTPUTS, ERRORS, RMSE, TR] = ANN(hiddenSizes,X,T,BP,ACT)
% ANN  Recebe 5 variáveis de entrada, a quantidade de neurônios e camadas
% da rede hiddenSizes, variáveis de entrada X, alvos da rede neural T, a
% função de ativação ACT e o algoritmo de backpropagation BP, e retorna a
% rede após o treinamento,  a saída calculada com a rede neural OUTPUTS, o
% erro em relação a saida da rede neural e o alvo ERRORS, a raiz do erro
% médio quadrático RMSE e, por fim, o registro do treinamento TR.
% 
%   Variáveis de entrada:
% 
%   hiddenSizes = Neurônios a serem usados na rede neural, seja com uma
%   ou mais camadas ocultas.Ex: [1 3].
%       
%   X = Uma matriz R x Q.
%   T = Uma matriz U x Q.
%   Onde:
%       Q = Número de amostras
%       R = Número de elementos de entrada
%       U = Numero de elementos de saída
% 
%   BP = Algorítimo de backpropagation. (Pradrão = 'trainlm')
%   http://bit.ly/MultilayerNeuralNetwork
% 
%   ACT = Função de ativação da rede neural.(Pradrão = 'tansig')
%   http://bit.ly/NeuralNetworkActivation
% 
%   SAÍDAS:
%   NET = Retorna a rede neural treinada como um objeto _network_ 

%   OUTPUTS = Retorna os resultados obtido pela rede neural como uma matriz
%   U x Q. 

%   ERRORS = Erro entre os valores alvo e os resultados da rede neural como
%   uma matriz U x Q

%   RMSE = Retorna raiz do erro médio quadrático como valor
%   numérico.

% 	TR = Retorna os dados do treinamento como _struct_ 

if nargin < 4
    BP = 'trainlm';
    ACT = 'tansig';
elseif nargin < 5
    ACT = 'tansig';
end

%% =========== ANN =============
%Inicializa a rede neural para regressão
NET = fitnet(hiddenSizes, BP);

%Define a função de ativação
for i = 1:length(hiddenSizes)
    NET.layers{i}.transferFcn = ACT;
end

%Para o caso da escolha da função de ativação sigmoid são definidos os
%valores da normalização dos dados para o intervalo de saida da função
%sigmoid [0 , 1].
if isequal(ACT,'logsig')
    saida = length(hiddenSizes) + 1;
    NET.inputs{1}.processParams{2}.ymin = 0;
    NET.outputs{saida}.processParams{2}.ymin = 0;
end
% Os valores padrão já estão dentro do intervalo para a função tangente
% hiperbólica [-1, 1] portanto não precosa ser mudado

%Treina a rede neural
[NET,TR] = train(NET,X,T,'useParallel','yes');

%Calculos dos valores obtidos pela rede neural
OUTPUTS = NET(X);
ERRORS = gsubtract(T,OUTPUTS);
RMSE = sqrt(mean((ERRORS).^2));

end