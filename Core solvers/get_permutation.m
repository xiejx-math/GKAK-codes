function [pi, perm] = get_sampling_order(m, iter, permutation, perm)

    switch permutation
        case 'IS'
            pi = 1:m;

        case 'SO'
            if iter == 1 || isempty(perm)
                perm = randperm(m);
            end
            pi = perm;

        case 'RR'
            pi = randperm(m);

        otherwise
            error('Unknown permutation strategy: %s', permutation);
    end
end
