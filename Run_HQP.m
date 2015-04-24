function [z,x] = Run_HQP(H,c,A,b)

% HILDRETH QUADRATIC PROGRAMMING ALGORITHM
%
% Solves QP Problems (via the QP Dual) in the form:
%
% Primal:
% Minimize 0.5x'Hx + c'x
% s.t. Ax <= b 
%
% Dual:
% Maximize e'lamda + 0.5lamda'*Dlamda - 0.5c'H^-1c
% s.t. lamda >= 0
%
% where,
% 
% 0.5c'H^-1c is a constant
% D = A*H^-1*A', the dual Quadratic coefficient matrix
% e = b + A*H_inv*c, the dual Linear coefficient matrix
%   
% Algorithm provided by user will be in the form:
%
% Minimize 0.5x'Hx + c'x
% s.t. Ax <= b 
%
% Dual objective problem is checked against KKT conditions
% An initial lambda basis is chosen ie: lamda(0,...,0) for m-variables
% Basis is built by solving, for each lamda, lamda(i) = max(0,w(i))
% Therefore, lamda values are all >= 0
%
% KKT condition check recurses until lamda at one iteration = previous  
%

    % SET MAX ITERATIONS
    maxIter = 250;
    
    iter = 1; % declaring iteration counter
    
    % adding non-negativity constraints to the A matrix
    A = cat(1,A,-eye(size(A,2)));
    
    % adding non-negativity constraints to the b matrix
    b = cat(1,b,zeros(size(A,2),1));

    
    % 1. DETERMINING INITIAL GLOBAL OPTIMAL SOLUTION, x
    % Preliminary check to see if x violates any constraints
    
    % x = -H^-1*c, since it rearranges back to -Hx=c
    % can use backslash operator
    x = -H\c;
    
    % checking if each constraint is satisfied by x
    % if all constraints are already satisfied, returns x as x*
    
    consFlag = true; % flag initialized as all constraints satisfied
    
    % for loop runs through each constraint and checks for violation
    % if violation found, consFlag changes to false
    
    ii = 1; % initializing index counter
    
    % consFlag changes to false if constraint violation found
    while ii < size(A,1)+1
        
        if(A(ii,:)*x > b(ii))
            consFlag = false;
            break;
            
        else ii = ii + 1;
    
        end
            
    end

    % if no constraint violations found, x = x* and z* = f(x)
    if consFlag ==  true
        z = 0.5.*x'*H*x + c'*x;
        return
    end    
        

    % 2. COMPUTATION OF THE DUAL OBJECTIVE FUNCTION COMPONENTS
    % given consFlag remained false, dual variable (lamda) is needed
        
    % quadratic lamda variable coefficients
    % given by D = A*H^-1*A', but let H^-1*A' = u, so D = A'*u
    % then H^-1*A' = u can be rearranged back to be Hu = A
    % therefore the backslash operator can be used
    % since actual inverse of H^-1 not needed
    D = A*(H\A');
    
    % linear lamda variable coefficients
    % given e = b + A*H^-1*c, but let H^-1*c = v, so e = b + A*v
    % then H^-1*c = v can be rearranged back to be Hv = A
    % therefore the backslash operator can be used
    % since actual inverse of H^-1 not needed
    e = b + A*(H\c);

    
    % 3. DETERMINING THE LAMDA SOLUTION VECTOR (LAGRANGE MATRIX)
    
    % solves KKTCond for lamda(i)
    % isolates for one lamda(i) sets all other lamda to basis values
    % KKTCond = lamda(i)*(dLagrange/dlamda(i)) = 0 for each lamda(i)
    % where each lamda(i) takes the value of max(0,-w/D(i,i))
    % this results in lamda(i) >= 0
    
    % recurses until new lambda basis = previous lambda basis
    % ie: lamda == lamdaCurr (current lamda)
    
    % initializing current lamda basis, ie: lamda0 = (0,...,0)
    lamdaCurr = zeros(size(e,1),1);
    
    % determines Lagrange variables iteratively one by one
    % iterates until algorithm reaches maximum number of iterations
    while iter < maxIter+1
        
        % setting previous lamda basis as current basis prior to iteration
        lamdaPrev = lamdaCurr;        

        for jj = 1:size(e,1)
        
            % calculating w for lamda(i) 
            w = (D(jj,:)*lamdaCurr-D(jj,jj)*lamdaCurr(jj,1)) + e(jj,1);
            
            % setting current lamda value to max(0,-a/D(i,i))
            lamdaCurr(jj,1) = max(0,-w/D(jj,jj));
             
        end
        
        % checking if new lamda basis is equal to previous basis
        % error allowance of 10^-9 within maximum number of iterations
        difference = lamdaCurr - lamdaPrev;
        
        if difference < 10^-9
            break;
        end 
       
    end

    % calculating x* value from optimal lamda basis
    % backslash operators used for same reason as explained in STEP 2
    x = -H\c - H\A'*lamdaCurr;
        
    % calculating z* value from optimal lamda basis
    z = 0.5.*x'*H*x + c'*x;

end

