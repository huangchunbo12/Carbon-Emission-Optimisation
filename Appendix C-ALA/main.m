% Artificial lemming algorithm: A novel bionic meta-heuristic technique for solving real-world engineering optimization problems
%                                                                                                     
%  Developed in MATLAB R2024a                                                                 
%                                                                                                     
%  Author :Yaning Xiao, Hao Cui, Ruba Abu Khurma, Pedro A. Castillo                                           
%                                                                                                     
%         e-Mail: xiaoyn2023@mail.sustech.edu.cn; cuihao@nefu.edu.cn; Rubaabukhurma82@gmail.com; pacv@ugr.es                                                                                                                                                                                          
%_______________________________________________________________________________________________
% You can simply define your cost function in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% MaxFEs = number of function evaluations
% N = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single numbers
% 2024-05-16
%______________________________________________________________________________________________

close all;clear all; clc
N=30;  
Function_name='F1'; 
[lb,ub,dim,fobj]=CEC2017(Function_name); % CEC2017
MaxFEs=30000;
[ala_score,ala_pos,ala_curve]=ALA(N,MaxFEs,lb,ub,dim,fobj);
display(['The best fitness of ALA is: ', num2str(ala_score)]);
figure('Position',[454   445   694   297]);
subplot(1,2,1);
func_plot_cec2017(Function_name);
title(Function_name)
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])
grid on
subplot(1,2,2);
semilogy(ala_curve,'Color','r','linewidth',2);
axis tight
title(Function_name)
xlabel('Iteration#')
ylabel('Best score obtained so far')
grid on
legend('ALA')

