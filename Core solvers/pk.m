function result_pk = pk(Dk, lisX, Xk)
    [~, p, ~] = size(Xk);
    [~, p2, ~] = size(lisX);
    t = p2/p ;
%     if k == 0
%         temp = zeros(k-jk, 1) ;
%     else
    temp = zeros(t, 1);
    for j = 1:t
        temp(j) = innerp(lisX(:,((j-1)*p+1):j*p, :)-Xk, Dk);
    end
%     end
    result_pk = temp;
end
