function [pi, perm] = get_permutation(m, iter, permutation, perm)
%GET_PERMUTATION Row order for cyclic tensor Kaczmarz sweeps.

switch upper(permutation)
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
