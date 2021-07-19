%��ȡѵ������
[f1,f2,f3] = textread('train_sample.txt' , '%f%f%f',200);
%����ֵ��һ��
[input,minI,maxI,output,minO,maxO] = premnmx([f1 , f2]',[f3]');
%����������
net = newff( minmax(input) , [10 1] , { 'logsig' 'purelin' } , 'traingdx' ) ; 
w1= net. iw{1, 1};%����㵽�м���Ȩֵ
b1 = net.b{1};%�м������Ԫ��ֵ
w2 = net.lw{2,1};%�м�㵽������Ȩֵ
b2= net. b{2};%��������Ԫ��ֵ
%����ѵ������
net.trainparam.show = 50 ;
net.trainparam.epochs = 800 ;
net.trainparam.goal = 1e-5 ;
net.trainParam.lr = 0.01 ;
%��ʼѵ��
net = train( net, input , output ) ;

%��ȡ��������
[t1 t2 t3 ] = textread('test_sample.txt' , '%f%f%f',200);
%�������ݹ�һ��
testInput = tramnmx ( [t1,t2]' , minI, maxI) ;
testOutput =[t3]';
%����
Y =sim( net , testInput );
%����һ��
testsim=postmnmx(Y,minI,maxI);
test1=sum(testsim);
%ͳ��ʶ����ȷ��
[s1 , s2] = size( testsim ) ;
error = 0 ;
for i = 1 : s2
    error=error+abs(test1(i)-testOutput(i));
end
sprintf('������� %3.3f%%', error/sum(testOutput,2)*100)

figure
plot(test1-testOutput,'linewidth',1.2)    %plot������ͼ  
title('���仯') 
xlabel('������������')
ylabel('���')
grid on 
