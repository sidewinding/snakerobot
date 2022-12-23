[f,v,n]=read_stl_file('x-y投影2.stl');                                                                                                                                                                      
hold on;
Ftotal=[0 0 0];
vtotal=0;
omega=2.5;
Fnum1=zeros(1,32);
Fnum2=zeros(1,32);
for i=1:length(f)
    N=[n(i,1) n(i,2) n(i,3)];
    V=[0, (-1.25*omega*sin(pi*v(f(i,1),1)/8)+vtotal)*cosd(0), (-1.25*omega*sin(pi*v(f(i,1),1)/8)+vtotal)*sind(0)];
    if norm(V)==0
        Funit=0;
    elseif v(f(i,1),1)<-15.99||v(f(i,1),1)>15.99
        Funit=0;
    elseif v(f(i,1),3)<-1.9
        Funit=0;
    else
        Nnorm=N/norm(N);
        Vnorm=V/norm(V);
        if dot(Nnorm,Vnorm)<=0
            Funit=0;
        elseif v(f(i,1),3)>=-1.7
            Funit=0;
        else
            e2=[Nnorm(1) Nnorm(2) 0]/norm([Nnorm(1) Nnorm(2) 0]);
            e3=[0 0 1];
            e1=cross(e2,e3);
            if Nnorm(3)>0
                beta=acos(dot(Nnorm,e2))-pi/2;
            else
                beta=-acos(dot(Nnorm,e2))+pi/2;
            end
            v1=dot(Vnorm,e1)*e1*norm(V);
            v23=V-v1;
            v23norm=v23/norm(v23);
            if v23(3)>0
                gamma=-acos(dot(v23norm,e2));
            else
                gamma=acos(dot(v23norm,e2));
            end
            az=0.206+0.169*cos(2*pi*(beta/pi))...
                +0.212*sin(2*pi*(beta/pi+gamma/(2*pi)))...
                +0.358*sin(2*pi*(gamma/(2*pi)))...
                +0.055*sin(2*pi*(-beta/pi+gamma/(2*pi)));
            ax=-0.124*cos(2*pi*(beta/pi+gamma/(2*pi)))...
                +0.253*cos(2*pi*gamma/(2*pi))...
                +0.007*cos(2*pi*(-beta/pi+gamma/(2*pi)))...
                +0.088*sin(2*pi*beta/pi);
            ay=-0.124+0.253+0.007;
            f23=norm(v23)/norm(V);
            f1=norm(v1)/norm(V);
            F1=f1;
            F2=(1+1.8/sqrt(tand(13.8)^2+(f23^2)))*f23/2.7480;
            ds=1/2*norm([(v(f(i,2),2)-v(f(i,1),2))*(v(f(i,3),3)-v(f(i,1),3))-(v(f(i,3),2)-v(f(i,1),2))*(v(f(i,2),3)-v(f(i,1),1)),...
                (v(f(i,2),3)-v(f(i,1),3))*(v(f(i,3),1)-v(f(i,1),1))-(v(f(i,3),3)-v(f(i,1),3))*(v(f(i,2),1)-v(f(i,1),1)),...
                (v(f(i,2),1)-v(f(i,1),1))*(v(f(i,3),2)-v(f(i,1),2))-(v(f(i,3),1)-v(f(i,1),1))*(v(f(i,2),2)-v(f(i,1),2))]);
            zeta=0.33;
            depth=-v(f(i,1),3)-1.7;
            FF1=-ax*F2*ds*zeta*e2*depth;
            if dot(e1,v1)>0    
                FF2=-ay*F1*ds*zeta*e1*depth;
            else
                FF2=ay*F1*ds*zeta*e1*depth;
            end
            FF3=az*F2*ds*zeta*e3*depth;
            Funit=real(FF1+FF2+FF3);
            num=ceil((v(f(i,1),1)+16));
            Fnum1(num)=Fnum1(num)+Funit(1);
            Fnum2(num)=Fnum2(num)+Funit(2);
            h=quiver3(v(f(i,1),1), v(f(i,1),2), v(f(i,1),3), -Funit(1)*30/ds,-Funit(2)*30/ds,-Funit(3)*30/ds,'c');
            set(h,'maxheadsize',2,'LineWidth',2);
            hold on;  
        end
    end
%     Fnum1=Fnum1-mean(Fnum1);
    Ftotal=Ftotal+Funit;
%     for j=1:16
%         xx=2*j;
%         yy=1.25*cos(pi*xx/8);
%         quiver3(xx,yy,0,Fnum1(j)*10,Fnum2(j),0,'r');
%         hold on
%     end
%     view(0,90);
    view(60,15);
end
            
            
        
        