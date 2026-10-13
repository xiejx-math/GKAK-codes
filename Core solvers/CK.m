function result_CK = CK(lstC)
    Lth = length(lstC);
    temp = zeros(Lth, Lth);
    if Lth > 0
        main_diag = [];
        for i = 1:Lth
            if i == 1
                main_diag = [main_diag, 1/lstC(i)];
            else 
                main_diag = [main_diag, 1/lstC(i-1)+1/lstC(i)];
            end
        end
        sub_diag = [];
        for i = 1:(Lth-1)
            sub_diag = [sub_diag, (-1)/lstC(i)] ;
        end
        if Lth > 1
            temp = diag(main_diag) + diag(sub_diag, 1) + diag(sub_diag, -1); 
        else
            temp = diag(main_diag);
        end
    end
    result_CK = temp ;
end