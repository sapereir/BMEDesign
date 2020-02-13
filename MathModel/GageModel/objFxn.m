function sumSqErrors = objFxn(x)
    A = x(1);
    K = x(2);
    C = x(3);
    B = x(4);
    v = x(5);
    Q = x(6);
    
    t = linspace(0,1000,1000);
    Pmodel = (K-A)./(C + Q.*exp(-B.*t)).^(1./v);
    Pexp = linspace(0,1000,1000);
    error = Pmodel - Pexp;
    errorSq = error.^2;
    sumSqErrors = sum(errorSq);
end