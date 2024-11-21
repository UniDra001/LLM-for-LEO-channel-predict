function [Ans_seq] = leo_satellite(angle_seq)
%LEO_SATELLITE 
%   根据轨道离心角，生成对应的地心坐标
    a=6871;
    e=0.02;
    E=angle_seq; % 
    x=a*(cos(E)-e);
    y=a*sqrt((1-e^2))*sin(E);
    z=0*E;
    DtoR=2*pi/360;
    A=98 * DtoR; % 升交点赤经角
    B=55*DtoR;
    C=pi/100;
    R3=[cos(A) -sin(A) 0;
        sin(A)  cos(A) 0;
        0        0     1;];
    R1=[1         0    0;
        0       cos(B)  -sin(B);
        0 sin(B) cos(B);];
    R2=[cos(C) -sin(C) 0;
        sin(C) cos(C) 0; 
        0  0  1;];
    R312=R3*R1*R2;
    Ans_seq=R312*[x;y;z;];
end

