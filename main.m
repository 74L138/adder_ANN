%读取训练数据
[f1,f2,f3] = textread('train_sample.txt' , '%f%f%f',200);
%特征值归一化
[input,minI,maxI,output,minO,maxO] = premnmx([f1 , f2]',[f3]');
%创建神经网络
net = newff( minmax(input) , [10 1] , { 'logsig' 'purelin' } , 'traingdx' ) ; 
w1= net. iw{1, 1};%输入层到中间层的权值
b1 = net.b{1};%中间各层神经元阈值
w2 = net.lw{2,1};%中间层到输出层的权值
b2= net. b{2};%输出层各神经元阈值
%设置训练参数
net.trainparam.show = 50 ;
net.trainparam.epochs = 800 ;
net.trainparam.goal = 1e-5 ;
net.trainParam.lr = 0.01 ;
%开始训练
net = train( net, input , output ) ;

%读取测试数据
[t1 t2 t3 ] = textread('test_sample.txt' , '%f%f%f',200);
%测试数据归一化
testInput = tramnmx ( [t1,t2]' , minI, maxI) ;
testOutput =[t3]';
%仿真
Y =sim( net , testInput );
%反归一化
testsim=postmnmx(Y,minI,maxI);
test1=sum(testsim);
%统计识别正确率
[s1 , s2] = size( testsim ) ;
error = 0 ;
for i = 1 : s2
    error=error+abs(test1(i)-testOutput(i));
end
sprintf('误差率是 %3.3f%%', error/sum(testOutput,2)*100)

figure
plot(test1-testOutput,'linewidth',1.2)    %plot函数作图  
title('误差变化') 
xlabel('测试样本组数')
ylabel('误差')
grid on 
