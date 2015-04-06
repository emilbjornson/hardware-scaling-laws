%This Matlab script can be used to generate Figure 6 in the article:
%
%Emil Björnson, Michail Matthaiou, Mérouane Debbah, "Massive MIMO with
%Non-Ideal Arbitrary Arrays: Hardware Scaling Laws and Circuit-Aware
%Design," IEEE Transactions on Wireless Communications, To appear.
%
%Download article: http://arxiv.org/abs/1409.0875
%
%This is version 1.0 (Last edited: 2015-04-06)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%
%Please note that the channels are generated randomly, thus the results
%will not be exactly the same as in the paper.

%Initialization
close all;
clear all;


%%Simulation parameters


rng('shuffle'); %Initiate the random number generators with a random seed
%%If rng('shuffle'); is not supported by your Matlab version, you can use
%%the following commands instead:
%randn('state',sum(100*clock));


%How many setups with different random user locations should be generated.
nbrOfSimulationSetups = 100;


%This parameters sets the set of different z-values that should be used
%along with the hardware imperfections; that is, if the level of
%imperfections should increase with the number of antennas or not.
%-1 : Ideal hardware
% 0 : Constant hardware imperfections
% Values larger than 0: The value used in the right hand side of the
% scaling law. The corresponding z values are assumed to be all the same
% (except z3=0 for a CO) and is computed below.
scalingExponents = [-1 0];


%Propagation environment
carrierFrequency = 2e9; %Carrier frequency 2 GHz
bandwidth = 10e6; %Hz in the LTE-like system
rho = 10^(5/10); %Power control parameter
noiseFloordBm = -174+10*log10(bandwidth); %Noise floor in dBm

%Parameters for hardware imperfections
LNA = 10^(2/10); %2 dB noise figure
ADC = 6; %6 bit quantization
zetaConstant = 1e-17; %Constant in phase noise process

%Compute hardware the key parameters for the different hardware imperfections
kappa2Original = 2^(-2*ADC) / (1- 2^(-2*ADC));
deltaOriginal = 4*pi^2*carrierFrequency^2/bandwidth*zetaConstant;
xiOriginal = LNA / (1- 2^(-2*ADC));


%Simulation scenario
BSs = 25; %Number of sites (one cell under study surrounded by 24 interfering cells)
N = 240; %Number of antennas per site
Ksite = 8; %Number of users per site
B = Ksite; %Pilot length

Tdifferent = [20:20:800 900:200:2500]; %Different lengths of the coherence block

shadowFadingStandardDeviation = 10^(5/10); %5 dB shadow fading standard deviation

%Define the size of the square cells in the simulation
intersiteDistance = 0.25; %Distance between the middle of two adjacent cells (in vertical or horizontal direction)
intersiteDistanceHalf = intersiteDistance/2; %Half the inter-cell distance
minimalUserDistance = 0.025; %Minimal distance between a user and the different locations where the base station might have its antennas (5 different locations)

Ktotal = nbrOfSimulationSetups * Ksite; %Total number of user location per cell (over all setups)


%Generate grid locations for the cells
locationHorizontal = repmat(-2*intersiteDistance : intersiteDistance : 2*intersiteDistance,[5 1]);
locVertical = locationHorizontal';
cellLocations = locationHorizontal(:) + 1i*locVertical(:); %Real part is horizontal coordinate and imaginary part is vertical coordinate
cellLocations = [cellLocations(13); cellLocations(1:12); cellLocations(14:end)]; %Put the middle cell at the first index, since it is the one under study


%Generate the relative deviation between the middle of the cell and the
%locations of the 4 arrays in the case of distributed antennas.
BSarrayOffsets = (intersiteDistance/4)*exp(1i*[pi/4 3*pi/4 5*pi/4 7*pi/4])';
A = length(BSarrayOffsets); %Number of distributed subarrays


%Generate user locations uniformly in the cells, but not too close to any
%of the locations where the sites might be.
UElocations = (rand(BSs*Ktotal,1)+1i*rand(BSs*Ktotal,1))*intersiteDistanceHalf; %Uniformly distributed in upper right part of the cell.
notReady = (abs(UElocations)<minimalUserDistance) | (abs(UElocations-BSarrayOffsets(4))<minimalUserDistance); %Check which of the user locations that satisfy the minimal distance constraints.
nbrToGenerate = sum(notReady(:)); %How many of the users need to be moved

%Go through the users until all of them have feasible locations
while nbrToGenerate>0
    UElocations(notReady) = (rand(nbrToGenerate,1)+1i*rand(nbrToGenerate,1))*intersiteDistanceHalf; %Redistribute users
    notReady = (abs(UElocations)<minimalUserDistance) | (abs(UElocations-BSarrayOffsets(4))<minimalUserDistance); %Check if the new locations are feasible
    nbrToGenerate = sum(notReady(:)); %How many of the users need to be moved
end

UElocations = reshape(UElocations,[BSs Ktotal]); %Make the user locations into matrix where the row gives the cell that the users belong to.

%Make sure that only one user in each "triangle" in each user setup
reference_case = repmat([true(BSs,nbrOfSimulationSetups) false(BSs,nbrOfSimulationSetups)],[1 Ksite/2]);
shouldswitch = (real(UElocations)>imag(UElocations) & reference_case)  | (real(UElocations)<imag(UElocations) & ~reference_case); %Check which of the users that are in the "wrong" triangle
UElocations(shouldswitch) = imag(UElocations(shouldswitch)) + 1i*real(UElocations(shouldswitch)); %Switch the vertical and horizontal coordiantes when necessary
rotations = kron(exp(1i*[0 pi/2 pi 3*pi/2]),ones(1,Ktotal/4)); %Prepare to rotate the users so there are
UElocations = UElocations .* repmat(rotations,[BSs 1]); %Rotate so that there are users in all the triangles
UElocations = UElocations + repmat(cellLocations,[1 Ktotal]); %Move the users to their real positions based on the cell that they belong to

%Calculate distances between each user and the arrays in each cell
distancesDistributed = zeros(A,BSs,BSs,Ktotal); %(a,j,l,k): Channel from subarray a of BS_j to UE_k in Cell l.

%Calculate the distances
for j = 1:BSs
    
    for m = 1:A
        distancesDistributed(m,j,:,:) = abs(UElocations - cellLocations(j)-BSarrayOffsets(m));
    end
    
end


%Create the DFT-based pilot matrix
FFTmatrix = fft(eye(B));
pilotSequencesDFTbased = FFTmatrix(:,1:Ksite);
Xtildepilot = sqrt(rho)*pilotSequencesDFTbased;


%Placeholders for storing of simulation results
rates_MRC_SOs_distributed_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Tdifferent),length(scalingExponents));
rates_MRC_CO_distributed_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Tdifferent),length(scalingExponents));

rates_MRC_SOs_distributed_analytical = zeros(1,Ksite,nbrOfSimulationSetups,length(Tdifferent),length(scalingExponents));
rates_MRC_CO_distributed_analytical = zeros(1,Ksite,nbrOfSimulationSetups,length(Tdifferent),length(scalingExponents));


%Go through all simulation setups (where the user locations are different)
for s = 1:nbrOfSimulationSetups
    
    %Extract indices of user locations that are used in the current setup
    userIndices = s:nbrOfSimulationSetups:Ktotal;
    
    
    %Compute all channel variances for distributed deployment
    shadowFadingRealizationsDistributed = randn(A,BSs,BSs,Ksite);
    lambdaDistributedOriginal = 10.^( -(128.1+37.6*log10(distancesDistributed(:,:,:,userIndices)) + shadowFadingStandardDeviation*shadowFadingRealizationsDistributed + noiseFloordBm)/10);
    
    lambdaDistributed = zeros(A,1,BSs,Ksite);
    
    for m = 1:BSs
        lambdaDistributed(:,1,m,:) = lambdaDistributedOriginal(:,1,m,:) ./ repmat(mean(lambdaDistributedOriginal(:,m,m,:),1),[A 1 1 1]);
    end
    
    
    %Go through the different scaling exponents from hardware scaling law
    %in Corollary 4. Simulation results will be generated for each of them.
    for ex = 1:length(scalingExponents)
        
        %The scaling exponents -1 corresponds to having ideal hardware
        if scalingExponents(ex)==-1
            kappa2 = 0;
            xi = 1;
            deltaCO = 0;
            deltaSOs = 0;
        end
        
        
        %Go through different lengths of the coherence block
        for tindex = 1:length(Tdifferent)
            
            %Output the progress of the simulation
            disp(['Setup: ' num2str(s) '/' num2str(nbrOfSimulationSetups) ', Hardware: ' num2str(ex) '/' num2str(length(scalingExponents)) ', Block: '  num2str(tindex) '/'  num2str(length(Tdifferent))]);
            
            %Extract out the current length of the coherence block
            T = Tdifferent(tindex);
            
            %Compute the level of hardware imperfections for the current
            %scaling exponent (zero or larger).
            if scalingExponents(ex) >= 0
                
                %Compute the largest common scaling exponent z=z_1=z_2=z_3
                %that can be handled with SLOs, if the right hand side in
                %Eq. (29) equals scalingExponents(ex).
                z = scalingExponents(ex)/(1+deltaOriginal*(T-B)/2);
                
                %kappa2 and xi are the same with a CO and SLOs
                kappa2 = kappa2Original*N^(z);
                xi = xiOriginal*N^(z);
                
                %delta is different for a CO and SLOs, since we cannot
                %increase it when having a CO
                deltaSOs = deltaOriginal*(1+log(N)*z);
                deltaCO = deltaOriginal;
                
            end
            
            
            %Compute X_lm in Eq. (12) and \bar{X}_lm in Eq. (14). These
            %equations are the same for a CO and SLOs, but we separate the
            %two cases since delta can be different if the hardware scaling
            %law is used.
            XlmCO = zeros(B,B,Ksite);
            XlmSOs = zeros(B,B,Ksite);
            XlmbarCO = zeros(B,B,Ksite);
            XlmbarSOs = zeros(B,B,Ksite);
            for m = 1:Ksite
                XlmCO(:,:,m) = (Xtildepilot(:,m)*Xtildepilot(:,m)').*toeplitz([1+kappa2 exp(-(1:B-1)*deltaCO/2)]);
                XlmbarCO(:,:,m) = (Xtildepilot(:,m)*Xtildepilot(:,m)').*toeplitz([1 exp(-(1:B-1)*deltaCO/2)]);
                XlmSOs(:,:,m) = (Xtildepilot(:,m)*Xtildepilot(:,m)').*toeplitz([1+kappa2 exp(-(1:B-1)*deltaSOs/2)]);
                XlmbarSOs(:,:,m) = (Xtildepilot(:,m)*Xtildepilot(:,m)').*toeplitz([1 exp(-(1:B-1)*deltaSOs/2)]);
            end
            
            
            %Compute D_{delta(t)} in Eq. (10) for a CO. For future
            %simplicity, we define its value at t=B and then define a
            %vector that gives the common scaling of all elements for t>B.
            %This is made possible the assumption that the pilots are
            %transmitted in the beginning of the coherence block.
            DdeltaB_CO = diag(exp(-(B-1:-1:0)*deltaCO/2)); %This is for t=B
            deltaTB_CO = exp(-(1:T-B)*deltaCO/2); %This is common scaling of all elements in D_{delta(t)} when t>B. More precisely, D_{delta(t)} = DdeltaB_CO * deltaTB_CO(t-B) for t = B+1,...T
            deltaTB2_CO = deltaTB_CO.^2; %For future use
            
            DdeltaB_SOs = diag(exp(-(B-1:-1:0)*deltaSOs/2));  %This is for t=B
            deltaTB_SOs = exp(-(1:T-B)*deltaSOs/2); %This is common scaling of all elements in D_{delta(t)} when t>B. More precisely, D_{delta(t)} = DdeltaB_CO * deltaTB_CO(t-B) for t = B+1,...T
            deltaTB2_SOs = deltaTB_SOs.^2; %For future use
            
            
            
            %Prealloacte placeholders for the different expectations in
            %Lemma 1, for co-located deployment.
            %These are used under the assumption that the pilot sequence is
            %sent the beginning of the coherence block.
            firstMoment_MRC_SOs_distributed_analytical = zeros(1,Ksite,T-B);
            firstMoment_MRC_CO_distributed_analytical = zeros(1,Ksite,T-B);
            secondMoments_MRC_SOs_distributed_analytical = zeros(1,Ksite,T-B);
            secondMoments_MRC_CO_distributed_analytical = zeros(1,Ksite,T-B);
            distortionTerm_MRC_SOs_distributed_analytical = zeros(1,Ksite,T-B);
            distortionTerm_MRC_CO_distributed_analytical = zeros(1,Ksite,T-B);
            filterNorm_MRC_SOs_distributedA_analytical = zeros(1,Ksite,T-B);
            filterNorm_MRC_CO_distributedA_analytical = zeros(1,Ksite,T-B);
            
            
            
            %We set the BS index j to 1 since we only compute the results
            %for the cell in the middle of Fig. 3.
            for j = 1
                
                I_A = eye(A); %Identity matrix of dimension A
                
                
                %Compute the matrix \tilde{\Psi}_j, defiend after Eq. (28),
                %for the distributed arrays of the special channel
                %covariance structure in Eq. (24).
                Psi_tildej_CO_distributed =  xi*eye(B*A);
                Psi_tildej_SOs_distributed =  xi*eye(B*A);
                for m = 1:Ksite
                    Psi_tildej_CO_distributed = Psi_tildej_CO_distributed + kron(XlmCO(:,:,m),diag(sum(lambdaDistributed(:,j,:,m),3)));
                    Psi_tildej_SOs_distributed = Psi_tildej_SOs_distributed + kron(XlmSOs(:,:,m),diag(sum(lambdaDistributed(:,j,:,m),3)));
                end
                
                
                %Go through all UEs in the middle cell
                for k = 1:Ksite
                    
                    %Compute the matrix in Eq. (37) for distributed
                    %deployments. Notice that this is the matrix that
                    %defines the LMMSE estimate in Eq. (9). We also
                    %precompute terms of the type in Eq. (27), namely
                    %\tilde{Psi}_j^{-1} ( D_{delta(t)} \tilde{x}_kl \kronecker e_{a_2}
                    Ajjk_CO_distributed = kron(Xtildepilot(:,k)'*DdeltaB_CO,diag(lambdaDistributed(:,j,j,k)))/Psi_tildej_CO_distributed;
                    Ajjk_SOs_distributed = kron(Xtildepilot(:,k)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,j,k)))/Psi_tildej_SOs_distributed;
                    
                    %Precompute terms of the type
                    %\tilde{Psi}_j^{-1} ( D_{delta(t)} \tilde{x}_kl \kronecker e_{a_2}
                    ejjk_CO_distributed = zeros(B*A,A);
                    ejjk_SOs_distributed = zeros(B*A,A);
                    for a=1:A
                        ejjk_CO_distributed(:,a) = Psi_tildej_CO_distributed\kron(DdeltaB_CO*Xtildepilot(:,k),I_A(:,a));
                        ejjk_SOs_distributed(:,a) = Psi_tildej_SOs_distributed\kron(DdeltaB_SOs*Xtildepilot(:,k),I_A(:,a));
                    end
                    
                    
                    
                    %Compute the expectations in Eqs. (20) and (21).
                    filterNorm_MRC_SOs_distributedA_analytical(j,k,:) = (N/A)*trace(Ajjk_SOs_distributed* kron(Xtildepilot(:,k)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_SOs,[1 1 T-B]);
                    filterNorm_MRC_CO_distributedA_analytical(j,k,:) = (N/A)*trace(Ajjk_CO_distributed* kron(Xtildepilot(:,k)'*DdeltaB_CO,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_CO,[1 1 T-B]);
                    firstMoment_MRC_SOs_distributed_analytical(j,k,:) = filterNorm_MRC_SOs_distributedA_analytical(j,k,:);
                    firstMoment_MRC_CO_distributed_analytical(j,k,:) = filterNorm_MRC_CO_distributedA_analytical(j,k,:);
                    
                    %Initialize the computation of the expectations in
                    %Eqs. (22) and (23).
                    sumLambdaAjlm = sum(sum(lambdaDistributed(:,j,:,:),3),4);
                    traceFormulaA_SOs =  (N/A)*trace(diag(sumLambdaAjlm)*Ajjk_SOs_distributed* kron(Xtildepilot(:,k)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_SOs,[1 1 T-B]);
                    traceFormulaA_CO =  (N/A)*trace(diag(sumLambdaAjlm)*Ajjk_CO_distributed* kron(Xtildepilot(:,k)'*DdeltaB_CO,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_CO,[1 1 T-B]);
                    distortionTerm_MRC_SOs_distributed_analytical(j,k,:) = kappa2*rho*traceFormulaA_SOs;
                    distortionTerm_MRC_CO_distributed_analytical(j,k,:) = kappa2*rho*traceFormulaA_CO;
                    secondMoments_MRC_SOs_distributed_analytical(j,k,:) = traceFormulaA_SOs;
                    secondMoments_MRC_CO_distributed_analytical(j,k,:) = traceFormulaA_CO;
                    
                    
                    %Go through all cells and UEs, in order to compute the
                    %sum of the interference terms and the distortion
                    %noise. Note that we  compute the results for different
                    %t-values at the same time, by multiplying with
                    %deltaTB which determines how the MRC filter changes
                    %over time. This gives the results for pilot
                    %transmission in the beginning of the coherence  block
                    for ll = 1:BSs
                        for mm = 1:Ksite
                            
                            %Compute the sum of the second order moments in
                            %Eq. (22) and the compute and sum up the
                            %expectation in Eq. (23).
                            secondMoments_MRC_SOs_distributed_analytical(j,k,:) = secondMoments_MRC_SOs_distributed_analytical(j,k,:) + (N/A)^2*trace(Ajjk_SOs_distributed* kron(Xtildepilot(:,mm)'*DdeltaB_SOs, diag(lambdaDistributed(:,j,ll,mm)))')^2 * reshape(deltaTB2_SOs.^2,[1 1 T-B]);
                            
                            for a1 = 1:A
                                
                                secondMoments_MRC_SOs_distributed_analytical(j,k,:) = secondMoments_MRC_SOs_distributed_analytical(j,k,:) + (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_SOs_distributed(:,a1)'*( kron(XlmSOs(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_SOs_distributed(:,a1) * reshape(deltaTB2_SOs,[1 1 T-B]) - (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_SOs_distributed(:,a1)'*( kron( DdeltaB_SOs*Xtildepilot(:,mm)*Xtildepilot(:,mm)'*DdeltaB_SOs ,I_A(:,a1)*I_A(:,a1)') )*ejjk_SOs_distributed(:,a1) * reshape(deltaTB2_SOs.^2,[1 1 T-B]);
                                secondMoments_MRC_CO_distributed_analytical(j,k,:) = secondMoments_MRC_CO_distributed_analytical(j,k,:) + (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_CO_distributed(:,a1)'*( kron(XlmCO(:,:,mm)-XlmbarCO(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_CO_distributed(:,a1) * reshape(deltaTB2_CO,[1 1 T-B]);
                                
                                distortionTerm_MRC_SOs_distributed_analytical(j,k,:) = distortionTerm_MRC_SOs_distributed_analytical(j,k,:) +  kappa2*rho* (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_SOs_distributed(:,a1)'*( kron(XlmSOs(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_SOs_distributed(:,a1) * reshape(deltaTB2_SOs,[1 1 T-B]);
                                distortionTerm_MRC_CO_distributed_analytical(j,k,:) = distortionTerm_MRC_CO_distributed_analytical(j,k,:) +  kappa2*rho* (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_CO_distributed(:,a1)'*( kron(XlmCO(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_CO_distributed(:,a1) * reshape(deltaTB2_CO,[1 1 T-B]);
                                
                                for a2 = 1:A
                                    
                                    secondMoments_MRC_CO_distributed_analytical(j,k,:) = secondMoments_MRC_CO_distributed_analytical(j,k,:) + (N/A)^2*lambdaDistributed(a1,j,j,k)*lambdaDistributed(a1,j,ll,mm)*lambdaDistributed(a2,j,j,k)*lambdaDistributed(a2,j,ll,mm)* ejjk_CO_distributed(:,a1)'*( kron(XlmbarCO(:,:,mm) ,I_A(:,a1)*I_A(:,a2)') )*ejjk_CO_distributed(:,a2) * reshape(deltaTB2_CO,[1 1 T-B]);
                                    
                                end
                                
                            end
                            
                        end
                        
                    end
                    
                end
                
            end
            
            
            %Compute the rates in Lemma 1 by using the analytic expressions
            %from Theorem 1 for each of the expectations
            rates_MRC_SOs_distributed_analytical(:,:,s,tindex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_SOs_distributed_analytical).^2 ./ ( rho* real(secondMoments_MRC_SOs_distributed_analytical )- rho*abs(firstMoment_MRC_SOs_distributed_analytical).^2 + real(distortionTerm_MRC_SOs_distributed_analytical) + xi*real(filterNorm_MRC_SOs_distributedA_analytical))),3)/T;
            rates_MRC_CO_distributed_analytical(:,:,s,tindex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_CO_distributed_analytical).^2 ./ ( rho* real(secondMoments_MRC_CO_distributed_analytical) - rho*abs(firstMoment_MRC_CO_distributed_analytical).^2 + real(distortionTerm_MRC_CO_distributed_analytical) + xi*real(filterNorm_MRC_CO_distributedA_analytical))),3)/T;
            
        end
        
    end
    
end



%Plot Figure 6 from the paper and compute the results with pilot
%transmission in the middle of the coherence block
figure(6); hold on; box on;

for ex=1:length(scalingExponents)
    
    %Ideal hardware: There are no phase drifts so it doesn't matter at
    %which channel uses that pilots are transmitted
    if scalingExponents(ex) == -1
        
        plot(Tdifferent,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,ex),3),2),1),[length(Tdifferent) 1])/(Ksite),'r-','LineWidth',1);
        
    else
        
        %Plot results with pilot transmission in the beginning of the
        %coherence block, including markers for the largest values
        rateSOs = reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,ex),3),2),1),[length(Tdifferent) 1])/(Ksite);
        [rateValSOs,indSOs] = max(rateSOs);
        
        rateCO = reshape(sum(sum(mean(rates_MRC_CO_distributed_analytical(:,:,:,:,ex),3),2),1),[length(Tdifferent) 1])/(Ksite);
        [rateValCO,indCO] = max(rateCO);
        
        plot(Tdifferent,rateSOs','k--','LineWidth',1);
        plot(Tdifferent,rateCO','b-.','LineWidth',1);
        
        plot(Tdifferent(indSOs),rateValSOs,'kd','LineWidth',1);
        plot(Tdifferent(indCO),rateValCO,'bd','LineWidth',1);
        
        
        %The results for pilot sequences in the beginning is transformed
        %to the case with pilot sequences in the middle of the coherence
        %block. This case can be seen as having a coherence block of length
        %2(T-B)+B = 2T-B, where the first T channel uses and the last T
        %channel uses (note the overlap!) are very similar  to the case
        %with pilot sequences in the beginning of the block (notice that
        %symmetry implies that pilots in the beginning or the end of the
        %block gives identical results). Consequently, with get an
        %achievable rate of 2*SumT/(2T-B), where SumT is the sum in Eq.
        %(19) for pilot sequences in the beginning of the block.
        rateSOs_middle = 2*Tdifferent.*rateSOs' ./ (2*Tdifferent-B);
        rateCO_middle = 2*Tdifferent.*rateCO' ./ (2*Tdifferent-B);
        
        plot(2*Tdifferent-B,rateSOs_middle,'k--','LineWidth',1);
        plot(2*Tdifferent-B,rateCO_middle,'b-.','LineWidth',1);
        
        [rateValSOs,indSOs] = max(rateSOs_middle);
        [rateValCO,indCO] = max(rateCO_middle);
        
        plot(2*Tdifferent(indSOs)-B,rateValSOs,'ko','LineWidth',1);
        plot(2*Tdifferent(indCO)-B,rateValCO,'bo','LineWidth',1);
        
    end
    
end

legend('Ideal Hardware','Non-Ideal Hardware: SLOs','Non-Ideal Hardware: CLO','Location','SouthWest')
xlabel('Coherence Block [channel uses]')
ylabel('Average Rate per UE [bit/channel use]')

axis([0 Tdifferent(end) 0 4]);
