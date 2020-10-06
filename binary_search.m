function alpha_stop=binary_search(U,D,V,u,v,delta)

%binary search for finding full step for Z \in it(B)

alpha_stop=0;alpha_max=1;
for i=1:10
    alpha=(alpha_max-alpha_stop)/2;
    [~,s,~]=svd_update(U,(1+alpha)*D,V,-alpha*delta*u,v);
    if sum(diag(s))<=delta %check if it's in feasible set
        alpha_stop=alpha;
    else
        alpha_max=alpha;
    end
end

end