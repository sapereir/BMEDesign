function [c, ceq] = nonlconst(x)
    A = x(1);
    K = x(2);
    C = x(3);
    B = x(4);
    v = x(5);
    Q = x(6);
    
    c = A;
    ceq = [];
end