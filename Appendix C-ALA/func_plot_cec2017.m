function func_plot_cec2017(func_name)

[lb,ub,dim,fobj]=CEC2017(func_name);

x=-100:2:100; y=x; % [-100,100]

L=length(x);
f=[];

for i=1:L
    for j=1:L
        if dim==2
            f(i,j)=fobj([x(i),y(j)]);
        else
            f(i,j)=fobj([x(i),y(j),zeros(1,dim-2)]);
        end
    end
end
surfc(x,y,f,'LineStyle','none');
end
