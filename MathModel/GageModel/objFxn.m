function sumSqErrors = objFxn(x,texp,Pexp)
    K = x(1);
    B = x(2);
    v = x(3);
    Q = x(4);
    
    sumSqErrors = 0;
    len = length(texp);
    
    %texp,Pext is a vector of all the experimental curves we are
    %analyzing with this optimization function
    for i = 1:len
        %[Pmax,Imax] = max(Pexp(i)); figure out how to truncate after
        %Maxiumum pressure is reached later
        Pmodel = (K)./(1 + Q.*exp(-B.*texp(i))).^(1./v);
        error = Pmodel - Pexp(i);
        errorSq = error.^2;
        sumSqErrors = sumSqErrors + sum(errorSq);
    end
end