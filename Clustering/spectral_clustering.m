function [C] = spectral_clustering(input, k, sigma)
    [rows, cols] = size(input);
    for i = 1:cols
        for j = 1:cols
            norm_sq = norm(input(:,i)-input(:,j))^2;
            num = (-1/(2*(sigma^2)))*norm_sq;
            A(i,j) = exp(num);
        
        end
    end
    
    for i = 1:cols
        sum = 0;
       for j = 1:cols
           sum = sum + A(i,j);
       end
       D(i,i) = sum; 
    end
    
    L = D - A
    
    [eig_vec, eig_val] = eig(L)
    
    val = diag(eig_val);
    
    [value, index] = sort(val);
    
    for i = 1:k
       if i > cols
           break
       else
           V(:,i) = eig_vec(:,index(i));
       end
    end
    
    C = kmeans(V,k);
    %x = input(1,:);
    %y = input(2,:);
    %scatter(x, y, 100, C , 'filled');
end

