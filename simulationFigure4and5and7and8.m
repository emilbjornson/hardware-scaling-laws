%This Matlab script can be used to generate all the simulation figures,
%except Figure 6, in the article:
%
%Emil Björnson, Michail Matthaiou, Mérouane Debbah, "Massive MIMO with
%Non-Ideal Arbitrary Arrays: Hardware Scaling Laws and Circuit-Aware
%Design," IEEE Transactions on Wireless Communications, vol. 14, 
%no. 8, pp. 4353-4368, August 2015.
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
%This number will have a large impact on the running time of the
%simulations, in particular when generating the figures that contains
%Monte Carlo simulations. It is highly recommended that you first run the
%code for a small number of setups or turn off all the Monte Carlo
%simulations, just to get a sense of the time it takes to generate a
%certain figure. As a rule-of-thumb, it takes minutes or hours to generate
%figures without Monte Carlo simulations, while it can takes days or weeks
%to get decent number of realizations when considering Monte Carlo
%simulations (in particular for the MMSE receiver in Fig. 8).
nbrOfSimulationSetups = 100;


%Select which of the figures that should be generated (4, 5, 7 or 8).
simulateFigure = 8;

%Initialize parameters for the selected figure
if simulateFigure == 4
    
    Nantennas = 4*[1 5 10:10:100]; %Different number of antennas per site (it is a multiple of 4, to make the distributed setup feasible).
    
    %Determine which of the colocated and distributed deployment scenarios from
    %Fig. 3 that will be simulated.
    runColocatedDeployment = true; %Should the co-located deployment be simulated (true/false)
    runDistributedDeployment = true; %Should the distributed deployment be simulated (true/false)
    
    %Should Monte Carlo simulations be ran?
    runMonteCarloSimulations = true; %Options: true or false
    runMonteCarlo_MRC = true; %Should we also run Monte Carlo simulations for MRC receiver filters (using Lemma 1)?
    runMonteCarlo_MRC_upper = true; %Should we also run Monte Carlo simulations for MRC receiver filters (by generalizing Lemma 1 based on Eq. (39) in [12])?
    runMonteCarlo_MMSE = false; %Should we also run Monte Carlo simulations for MMSE receiver filters?
    nbrOfMonteCarloRealizations = 100; %How many Monte Carlo simulations should be ran for each setup (the memory and number of antennas puts a limit to this parameter, but note that we use the Monte Carlo simulations to estimate scalars so we don't need extremly large values).
    
    
    %This parameters sets the set of different z-values that should be used
    %along with the hardware imperfections; that is, if the level of
    %imperfections should increase with the number of antennas or not.
    %-1 : Ideal hardware
    % 0 : Constant hardware imperfections
    % Values larger than 0: The value used in the right hand side of the
    % scaling law. The corresponding z values are assumed to be all the same
    % (except z3=0 for a CO) and is computed below.
    scalingExponents = [-1 0];
    
    %This parameter decides which type of pilot sequences that should be
    %used.
    %1 : Spatially orthogonal DFT-based pilot sequences
    %2 : Temporally orthogonal pilot sequence
    pilotBooks = [1 1];
    
    
    %Propagation environment
    carrierFrequency = 2e9; %Carrier frequency 2 GHz
    bandwidth = 10e6; %Hz in the LTE-like system
    rho = 10^(5/10); %Power control parameter
    noiseFloordBm = -174+10*log10(bandwidth); %Noise floor in dBm
    
    
    %Parameters for hardware imperfections
    LNA = 10^(2/10); %2 dB noise figure
    ADC = 6; %6 bit quantization
    zetaConstant = 1e-17; %Constant in phase noise process
    
    %Compute the key parameters for the different hardware imperfections
    kappa2Original = 2^(-2*ADC) / (1- 2^(-2*ADC));
    deltaOriginal = 4*pi^2*carrierFrequency^2/bandwidth*zetaConstant;
    xiOriginal = LNA / (1- 2^(-2*ADC));
    

elseif simulateFigure == 5
    Nantennas = ceil(10.^(0.4:0.4:6)/4)*4; %Different number of antennas per site (it is a multiple of 4, to make the distributed setup feasible).
    
    %Determine which of the colocated and distributed deployment scenarios from
    %Fig. 3 that will be simulated.
    runColocatedDeployment = false; %Should the co-located deployment be simulated (true/false)
    runDistributedDeployment = true; %Should the distributed deployment be simulated (true/false)
    
    %Should Monte Carlo simulations be ran?
    runMonteCarloSimulations = false; %Options: true or false
    runMonteCarlo_MRC = false; %Should we also run Monte Carlo simulations for MRC receiver filters?
    runMonteCarlo_MRC_upper = false; %Should we also run Monte Carlo simulations for MRC receiver filters (by generalizing Lemma 1 based on Eq. (39) in [12])?
    runMonteCarlo_MMSE = false; %Should we also run Monte Carlo simulations for MMSE receiver filters?
    nbrOfMonteCarloRealizations = 100; %How many Monte Carlo simulations should be ran for each setup (the memory and number of antennas puts a limit to this parameter, but note that we use the Monte Carlo simulations to estimate scalars so we don't need extremly large values).
    
    
    %This parameters sets the set of different z-values that should be used
    %along with the hardware imperfections; that is, if the level of
    %imperfections should increase with the number of antennas or not.
    %-1 : Ideal hardware
    % 0 : Constant hardware imperfections
    % Values larger than 0: The value used in the right hand side of the
    % scaling law. The corresponding z values are assumed to be all the same
    % (except z3=0 for a CO) and is computed below.
    scalingExponents = [-1 0 -1 0];
    
    %This parameter decides which type of pilot sequences that should be
    %used along with each of the scaling expontents defined above.
    %1 : Spatially orthogonal DFT-based pilot sequences
    %2 : Temporally orthogonal pilot sequence
    pilotBooks = [1 1 2 2];
    
    %Propagation environment
    carrierFrequency = 2e9; %Carrier frequency 2 GHz
    bandwidth = 10e6; %Hz in the LTE-like system
    rho = 10^(5/10); %Power control parameter
    noiseFloordBm = -174+10*log10(bandwidth); %Noise floor in dBm
    
    %Parameters for hardware imperfections
    LNA = 10^(2/10); %2 dB noise figure
    ADC = 6; %6 bit quantization
    zetaConstant = 1e-17; %Constant in phase noise process
    
    %Compute the key parameters for the different hardware imperfections
    kappa2Original = 2^(-2*ADC) / (1- 2^(-2*ADC));
    deltaOriginal = 4*pi^2*carrierFrequency^2/bandwidth*zetaConstant;
    xiOriginal = LNA / (1- 2^(-2*ADC));
    
    
elseif simulateFigure == 7
    
    Nantennas = 4*[1 5 10:10:40 60:20:100]; %Different number of antennas per site (it is a multiple of 4, to make the distributed setup feasible).
    
    %Determine which of the colocated and distributed deployment scenarios from
    %Fig. 3 that will be simulated.
    runColocatedDeployment = false; %Should the co-located deployment be simulated (true/false)
    runDistributedDeployment = true; %Should the distributed deployment be simulated (true/false)
    
    %Should Monte Carlo simulations be ran?
    runMonteCarloSimulations = false; %Options: true or false
    runMonteCarlo_MRC = false; %Should we also run Monte Carlo simulations for MRC receiver filters?
    runMonteCarlo_MRC_upper = false; %Should we also run Monte Carlo simulations for MRC receiver filters (by generalizing Lemma 1 based on Eq. (39) in [12])?
    runMonteCarlo_MMSE = false; %Should we also run Monte Carlo simulations for MMSE receiver filters?
    nbrOfMonteCarloRealizations = 1000; %How many Monte Carlo simulations should be ran for each setup (the memory and number of antennas puts a limit to this parameter, but note that we use the Monte Carlo simulations to estimate scalars so we don't need extremly large values).
    
    %This parameters sets the set of different z-values that should be used
    %along with the hardware imperfections; that is, if the level of
    %imperfections should increase with the number of antennas or not.
    %-1 : Ideal hardware
    % 0 : Constant hardware imperfections
    % Values larger than 0: The value used in the right hand side of the
    % scaling law. The corresponding z values are assumed to be all the same
    % (except z3=0 for a CO) and is computed below.
    scalingExponents = [-1 0 0.5 1];
    
    %This parameter decides which type of pilot sequences that should be
    %used.
    %1 : Spatially orthogonal DFT-based pilot sequences
    %2 : Temporally orthogonal pilot sequence
    pilotBooks = [1 1 1 1];
    
    
    %Propagation environment
    carrierFrequency = 2e9; %Carrier frequency 2 GHz
    bandwidth = 10e6; %Hz in the LTE-like system
    rho = 10^(15/10); %Power control parameter
    noiseFloordBm = -174+10*log10(bandwidth); %Noise floor in dBm
    
    %Set the key parameters for the different hardware imperfections
    kappa2Original = 0.05^2;
    deltaOriginal = 7e-5;
    xiOriginal = 3;
    
    
elseif simulateFigure == 8
    
    Nantennas = 4*[1 5 10:10:40 60:20:100]; %Different number of antennas per site (it is a multiple of 4, to make the distributed setup feasible).
    
    %Determine which of the colocated and distributed deployment scenarios from
    %Fig. 3 that will be simulated.
    runColocatedDeployment = false; %Should the co-located deployment be simulated (true/false)
    runDistributedDeployment = true; %Should the distributed deployment be simulated (true/false)
    
    %Should Monte Carlo simulations be ran?
    runMonteCarloSimulations = true; %Options: true or false
    runMonteCarlo_MRC = false; %Should we also run Monte Carlo simulations for MRC receiver filters?
    runMonteCarlo_MRC_upper = false; %Should we also run Monte Carlo simulations for MRC receiver filters (by generalizing Lemma 1 based on Eq. (39) in [12])?
    runMonteCarlo_MMSE = true; %Should we also run Monte Carlo simulations for MMSE receiver filters?
    nbrOfMonteCarloRealizations = 200; %How many Monte Carlo simulations should be ran for each setup (the memory and number of antennas puts a limit to this parameter, but note that we use the Monte Carlo simulations to estimate scalars so we don't need extremly large values).
    
    
    %This parameters sets the set of different z-values that should be used
    %along with the hardware imperfections; that is, if the level of
    %imperfections should increase with the number of antennas or not.
    %-1 : Ideal hardware
    % 0 : Constant hardware imperfections
    % Values larger than 0: The value used in the right hand side of the
    % scaling law. The corresponding z values are assumed to be all the same
    % (except z3=0 for a CO) and is computed below.
    scalingExponents = [-1 0 0.5 1];
    
    %This parameter decides which type of pilot sequences that should be
    %used.
    %1 : Spatially orthogonal DFT-based pilot sequences
    %2 : Temporally orthogonal pilot sequence
    pilotBooks = [1 1 1 1];
    
    
    %Propagation environment
    carrierFrequency = 2e9; %Carrier frequency 2 GHz
    bandwidth = 10e6; %Hz in the LTE-like system
    rho = 10^(15/10); %Power control parameter
    noiseFloordBm = -174+10*log10(bandwidth); %Noise floor in dBm
    
    %Set the key parameters for the different hardware imperfections
    kappa2Original = 0.05^2;
    deltaOriginal = 7e-5;
    xiOriginal = 3;
    
    
end



%Simulation scenario
BSs = 25; %Number of sites (one cell under study surrounded by 24 interfering cells)

maxN = max(Nantennas); %Extract out the largest number of antennas per site
Ksite = 8; %Number of users per site

B = Ksite; %Pilot length
T = 500; %Coherence time
shadowFadingStandardDeviation = 5; %5 dB shadow fading standard deviation

%Define the size of the square cells in the simulation
intersiteDistance = 0.25; %Distance (in km) between the middle of two adjacent cells (in vertical or horizontal direction)
intersiteDistanceHalf = intersiteDistance/2; %Half the inter-cell distance
minimalUserDistance = 0.025; %Minimal distance (in km) between a user and the different locations where the base station might have its antennas (5 different locations)

Ktotal = nbrOfSimulationSetups * Ksite; %Total number of user location per cell (over all setups)


%Generate grid locations for the cells
locationHorizontal = repmat(-2*intersiteDistance : intersiteDistance : 2*intersiteDistance,[5 1]);
locationVertical = locationHorizontal';
cellLocations = locationHorizontal(:) + 1i*locationVertical(:); %Real part is horizontal coordinate and imaginary part is vertical coordinate
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
distancesColocated = zeros(BSs,BSs,Ktotal); %(j,l,k): Channel from BS_j to UE_k in Cell l.
distancesDistributed = zeros(A,BSs,BSs,Ktotal); %(a,j,l,k): Channel from subarray a of BS_j to UE_k in Cell l.

%Calculate the distances
for j = 1:BSs
    
    distancesColocated(j,:,:) = abs(UElocations - cellLocations(j));
    
    for m = 1:A
        distancesDistributed(m,j,:,:) = abs(UElocations - cellLocations(j)-BSarrayOffsets(m));
    end
    
end


%Generate the two types of pilot matrices from Example 1
identityMatrix = eye(B);
pilotSequencesIdentityBased = identityMatrix(:,1:Ksite); %Temporally orthogonal pilot sequences from Eq. (6).
FFTmatrix = fft(eye(B));
pilotSequencesDFTbased = FFTmatrix(:,1:Ksite); %Spatially orthogonal pilot sequences from Eq. (7).



%Placeholders for storing of simulation results (co-located deployment)
if runColocatedDeployment == true
    
    %Based on closed-form expressions for MRC receiver filters, using Lemma 1 and Theorem 2
    rates_MRC_SOs_colocated_analytical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    rates_MRC_CO_colocated_analytical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    
    %Asymptotic rates as N->infinity for MRC receiver filters, using Lemma 1 and Theorem 2
    rates_MRC_SOs_colocated_asymptotics = zeros(1,Ksite,nbrOfSimulationSetups,length(scalingExponents));
    rates_MRC_CO_colocated_asymptotics = zeros(1,Ksite,nbrOfSimulationSetups,length(scalingExponents));
    
    %Used for Monte Carlo simulations with MRC receiver filters, using Lemma 1
    if runMonteCarlo_MRC == true
        rates_MRC_SOs_colocated_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
        rates_MRC_CO_colocated_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    end
    
    %Used for Monte Carlo simulations with MRC receiver filters, by generalizing Lemma 1 to condition on channel estimates
    if runMonteCarlo_MRC_upper == true
        rates_MRC_upper_colocated_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    end
    
    %Used for Monte Carlo simulations with MMSE receiver filters
    if runMonteCarlo_MMSE == true
        rates_MMSE_SOs_colocated_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
        rates_MMSE_CO_colocated_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    end
    
end

%Placeholders for storing of simulation results (distributed deployment)
if runDistributedDeployment == true
    
    %Based on closed-form expressions for MRC receiver filters, using Lemma 1 and Theorem 2
    rates_MRC_SOs_distributed_analytical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    rates_MRC_CO_distributed_analytical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    
    %Asymptotic rates as N->infinity for MRC receiver filters, using Lemma 1 and Theorem 2
    rates_MRC_SOs_distributed_asymptotics = zeros(1,Ksite,nbrOfSimulationSetups,length(scalingExponents));
    rates_MRC_CO_distributed_asymptotics = zeros(1,Ksite,nbrOfSimulationSetups,length(scalingExponents));
    
    %Used for Monte Carlo simulations with MRC receiver filters, using Lemma 1
    if runMonteCarlo_MRC == true
        rates_MRC_SOs_distributed_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
        rates_MRC_CO_distributed_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    end
    
    %Used for Monte Carlo simulations with MRC receiver filters, by generalizing Lemma 1 to condition on channel estimates
    if runMonteCarlo_MRC_upper == true
        rates_MRC_upper_distributed_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    end
    
    %Used for Monte Carlo simulations with MMSE receiver filters
    if runMonteCarlo_MMSE == true
        rates_MMSE_SOs_distributed_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
        rates_MMSE_CO_distributed_numerical = zeros(1,Ksite,nbrOfSimulationSetups,length(Nantennas),length(scalingExponents));
    end
    
end




%Go through all simulation setups (where the user locations are different)
for s = 1:nbrOfSimulationSetups
    
    %Extract indices of user locations that are used in the current setup
    userIndices = s:nbrOfSimulationSetups:Ktotal;
    
    
    %Compute all channel variances for colocated deployment and scale
    %based on the channel-inversion power control. Note that the distances 
    %are measured in kilometer and not meter as in Eq. (33) of the paper
    if runColocatedDeployment == true
        shadowFadingRealizationsColocated = randn(BSs,BSs,Ksite);
        lambdaColocatedOriginal = 10.^( -(128.1+37.6*log10(distancesColocated(:,:,userIndices)) + shadowFadingStandardDeviation*shadowFadingRealizationsColocated + noiseFloordBm)/10);
        
        lambdaColocated = zeros(1,BSs,Ksite);
        
        for m = 1:BSs
            lambdaColocated(1,m,:) = lambdaColocatedOriginal(1,m,:) ./ lambdaColocatedOriginal(m,m,:);
        end
        
    end
    
    %Compute all channel variances for distributed deployment and scale
    %based on the channel-inversion power control. Note that the distances 
    %are measured in kilometer and not meter as in Eq. (33) of the paper
    if runDistributedDeployment == true
        shadowFadingRealizationsDistributed = randn(A,BSs,BSs,Ksite);
        lambdaDistributedOriginal = 10.^( -(128.1+37.6*log10(distancesDistributed(:,:,:,userIndices)) + shadowFadingStandardDeviation*shadowFadingRealizationsDistributed + noiseFloordBm)/10);
        
        lambdaDistributed = zeros(A,1,BSs,Ksite);
        
        for m = 1:BSs
            lambdaDistributed(:,1,m,:) = lambdaDistributedOriginal(:,1,m,:) ./ repmat(mean(lambdaDistributedOriginal(:,m,m,:),1),[A 1 1 1]);
        end
    end
    
    
    %Generate random channel realizations
    if runMonteCarloSimulations == true
        
        %Compute the Rayleigh fading channel realizations for the maximum
        %number of antennas, before the channel variances have been applied.
        channelRealizations = (randn(maxN,1,BSs,Ksite,nbrOfMonteCarloRealizations)+1i*randn(maxN,1,BSs,Ksite,nbrOfMonteCarloRealizations))/sqrt(2);
        
        %Scale the channel realizations using the channel variances for colocated deployment
        if runColocatedDeployment == true
            Hcolocated = repmat(reshape(sqrt(lambdaColocated(1,:,:)),[1 1 BSs Ksite]),[maxN 1 1 1 nbrOfMonteCarloRealizations]) .* channelRealizations;
        end
        
        %Scale the channel realizations using the channel variances for distributed deployment
        if runDistributedDeployment == true
            Hdistributed = zeros(size(channelRealizations));
            
            for a = 1:A
                Hdistributed(1+(a-1)*(maxN/A):a*(maxN/A),:,:,:) = repmat(reshape(sqrt(lambdaDistributed(a,1,:,:)),[1 1 BSs Ksite]),[maxN/A 1 1 nbrOfMonteCarloRealizations]) .* channelRealizations(1+(a-1)*(maxN/A):a*(maxN/A),:,:,:);
            end
        end
        
        %Remove the initial channel realization matrix, since it may take
        %up a lot of memory.
        clear channelRealizations;
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
        
        
        %Go through the different number of antenans that are considered
        for nIndex = 1:length(Nantennas)
            
            %Output the progress of the simulation
            disp(['Setup: ' num2str(s) '/' num2str(nbrOfSimulationSetups) ', Hardware: ' num2str(ex) '/' num2str(length(scalingExponents)) ', Antennas: '  num2str(nIndex) '/'  num2str(length(Nantennas))]);
            
            
            %Extract the current number of antennas
            N = Nantennas(nIndex);
            
            %Extract the indices of the N elements in the channel
            %realizations that are used in current case. These are not the
            %first N elements, since the channel covariance matrices are
            %generated as in Eq. (24) for the distributed deployment.
            if runMonteCarloSimulations == true
                activeAntennas = zeros(N,1);
                for a = 1:A
                    activeAntennas(1+(a-1)*(N/A):a*(N/A)) = 1+(a-1)*(maxN/A):(N/A)+(a-1)*(maxN/A);
                end
            end
            
            
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
            
            
            %Extract out the pilot sequence matrix that is used in the
            %current iteration.
            if pilotBooks(ex) == 1
                Xtildepilot = sqrt(rho)*pilotSequencesDFTbased; %Spatially orthogonal pilots
            elseif pilotBooks(ex) == 2
                Xtildepilot = sqrt(rho)*pilotSequencesIdentityBased; %Temporally orthogonal pilots
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
            if runColocatedDeployment == true
                
                firstMoment_MRC_SOs_colocated_analytical = zeros(1,Ksite,T-B);
                firstMoment_MRC_CO_colocated_analytical = zeros(1,Ksite,T-B);
                secondMoments_MRC_SOs_colocated_analytical = zeros(1,Ksite,T-B);
                secondMoments_MRC_CO_colocated_analytical = zeros(1,Ksite,T-B);
                distortionTerm_MRC_SOs_colocated_analytical = zeros(1,Ksite,T-B);
                distortionTerm_MRC_CO_colocated_analytical = zeros(1,Ksite,T-B);
                filterNorm_MRC_SOs_colocated_analytical = zeros(1,Ksite,T-B);
                filterNorm_MRC_CO_colocated_analytical = zeros(1,Ksite,T-B);
                
                if runMonteCarlo_MRC == true
                    
                    firstMoment_MRC_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MRC_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MRC_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MRC_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    
                    firstMoment_MRC_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MRC_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MRC_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MRC_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    
                end
                
                if runMonteCarlo_MMSE == true
                    
                    firstMoment_MMSE_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MMSE_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MMSE_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MMSE_SOs_colocated_numerical = zeros(1,Ksite,T-B);
                    
                    firstMoment_MMSE_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MMSE_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MMSE_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MMSE_CO_colocated_numerical = zeros(1,Ksite,T-B);
                    
                end
                
            end
            
            
            %Prealloacte placeholders for the different expectations in
            %Lemma 1, for distributed deployment.
            %These are used under the assumption that the pilot
            %sequence is sent the beginning of the coherence block.
            if runDistributedDeployment == true
                
                firstMoment_MRC_SOs_distributed_analytical = zeros(1,Ksite,T-B);
                firstMoment_MRC_CO_distributed_analytical = zeros(1,Ksite,T-B);
                secondMoments_MRC_SOs_distributed_analytical = zeros(1,Ksite,T-B);
                secondMoments_MRC_CO_distributed_analytical = zeros(1,Ksite,T-B);
                distortionTerm_MRC_SOs_distributed_analytical = zeros(1,Ksite,T-B);
                distortionTerm_MRC_CO_distributed_analytical = zeros(1,Ksite,T-B);
                filterNorm_MRC_SOs_distributed_analytical = zeros(1,Ksite,T-B);
                filterNorm_MRC_CO_distributed_analytical = zeros(1,Ksite,T-B);
                
                if runMonteCarlo_MRC == true
                    
                    firstMoment_MRC_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MRC_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MRC_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MRC_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    
                    firstMoment_MRC_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MRC_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MRC_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MRC_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    
                end
                
                if runMonteCarlo_MMSE == true
                    
                    firstMoment_MMSE_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MMSE_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MMSE_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MMSE_SOs_distributed_numerical = zeros(1,Ksite,T-B);
                    
                    firstMoment_MMSE_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    secondMoments_MMSE_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    distortionTerm_MMSE_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    filterNorm_MMSE_CO_distributed_numerical = zeros(1,Ksite,T-B);
                    
                end
                
            end
            
            
            
            %We set the BS index j to 1 since we only compute the results
            %for the cell in the middle of Fig. 3.
            for j = 1
                
                I_A = eye(A); %Identity matrix of dimension A
                
                
                %Compute the matrix \Omega_j in (18), which is used for
                %co-located deployments.
                if runColocatedDeployment == true
                    Omegaj_CO = xi*eye(B);
                    Omegaj_SOs = xi*eye(B);
                    for m = 1:Ksite
                        Omegaj_CO = Omegaj_CO + sum(lambdaColocated(j,:,m)) * XlmCO(:,:,m);
                        Omegaj_SOs = Omegaj_SOs + sum(lambdaColocated(j,:,m)) * XlmSOs(:,:,m);
                    end
                end
                
                
                %Compute the matrix \tilde{\Psi}_j, defiend after Eq. (28),
                %for the distributed arrays of the special channel
                %covariance structure in Eq. (24).
                if runDistributedDeployment == true
                    Psi_tildej_CO_distributed =  xi*eye(B*A);
                    Psi_tildej_SOs_distributed =  xi*eye(B*A);
                    for m = 1:Ksite
                        Psi_tildej_CO_distributed = Psi_tildej_CO_distributed + kron(XlmCO(:,:,m),diag(sum(lambdaDistributed(:,j,:,m),3)));
                        Psi_tildej_SOs_distributed = Psi_tildej_SOs_distributed + kron(XlmSOs(:,:,m),diag(sum(lambdaDistributed(:,j,:,m),3)));
                    end
                end
                
                
                %Generate random realization of the phase drifts, distortion
                %noises, and receiver noises (defined in Section II).
                if runMonteCarloSimulations == true
                    
                    %Phase drift realizations for a CO and SOs
                    phijCO = cumsum(sqrt(deltaCO)*randn(1,T,nbrOfMonteCarloRealizations),2);
                    phijSOs = cumsum(sqrt(deltaSOs)*randn(N,T,nbrOfMonteCarloRealizations),2);
                    
                    %Realizations of the distortion noise for co-located
                    %and distributed deployments.
                    if runColocatedDeployment == true
                        Upsilon_colocated_sqrtj = reshape(sqrt(kappa2*rho*sum(sum(abs(Hcolocated(activeAntennas,j,:,:,:)).^2,3),4)),[N 1 nbrOfMonteCarloRealizations]);
                        upsilon_colocatedj = repmat(Upsilon_colocated_sqrtj/sqrt(2),[1 T 1]) .* (randn(N,T,nbrOfMonteCarloRealizations)+1i*randn(N,T,nbrOfMonteCarloRealizations));
                    end
                    
                    if runDistributedDeployment == true
                        Upsilon_distributed_sqrtj = reshape(sqrt(kappa2*rho*sum(sum(abs(Hdistributed(activeAntennas,j,:,:,:)).^2,3),4)),[N 1 nbrOfMonteCarloRealizations]);
                        upsilon_distributedj = repmat(Upsilon_distributed_sqrtj/sqrt(2),[1 T 1]) .* (randn(N,T,nbrOfMonteCarloRealizations)+1i*randn(N,T,nbrOfMonteCarloRealizations));
                    end
                    
                    %Realizations of the receiver noise
                    etaj = sqrt(xi/2)*(randn(N,B,nbrOfMonteCarloRealizations)+1i*randn(N,B,nbrOfMonteCarloRealizations));
                    
                end
                
                
                %Go through all realizations in the Monte Carlo simulations.
                %The analytical expressions are computed when r=1.
                for r = 1:nbrOfMonteCarloRealizations
                    
                    
                    if runMonteCarloSimulations == true
                        
                        if runColocatedDeployment == true
                            Yj_SOs_colocated = exp(1i*phijSOs(:,1:B,r)).*(reshape(sum(Hcolocated(activeAntennas,j,:,:,r),3),[N Ksite]) * transpose(Xtildepilot)) + upsilon_colocatedj(:,1:B,r) + etaj(:,:,r);
                            Yj_CO_colocated = exp(1i*repmat(phijCO(1,1:B,r),[N 1])).*(reshape(sum(Hcolocated(activeAntennas,j,:,:,r),3),[N Ksite]) * transpose(Xtildepilot)) + upsilon_colocatedj(:,1:B,r) + etaj(:,:,r);
                        end
                        
                        if runDistributedDeployment == true
                            Yj_SOs_distributed = exp(1i*phijSOs(:,1:B,r)).*(reshape(sum(Hdistributed(activeAntennas,j,:,:,r),3),[N Ksite]) * transpose(Xtildepilot)) + upsilon_distributedj(:,1:B,r) + etaj(:,:,r);
                            Yj_CO_distributed = exp(1i*repmat(phijCO(1,1:B,r),[N 1])).*(reshape(sum(Hdistributed(activeAntennas,j,:,:,r),3),[N Ksite]) * transpose(Xtildepilot)) + upsilon_distributedj(:,1:B,r) + etaj(:,:,r);
                        end
                        
                        %Prepare the MMSE receive filters in Eq. (33) by
                        %computing the matrix that is inverted. It is
                        %denoted by Zj.
                        if runMonteCarlo_MMSE == true || (runMonteCarlo_MRC_upper == true && scalingExponents(ex)==-1)
                            
                            %Initialize Zj for different deployments.
                            if runColocatedDeployment == true
                                Zj_SOs_colocated = xi*eye(N);
                                Zj_CO_colocated = xi*eye(N);
                            end
                            
                            if runDistributedDeployment == true
                                Zj_SOs_distributed = xi*eye(N);
                                Zj_CO_distributed = xi*eye(N);
                            end
                            
                            %Go through all BSs and their UEs, in order to
                            %compute the sum expressions in Eq. (33).
                            for ll = 1:BSs
                                for mm = 1:Ksite
                                    
                                    %Consider co-located deployments
                                    if runColocatedDeployment == true
                                        
                                        %Compute the channel estimate
                                        %\hat{h}_{jlm} for SOs using Theorem 1
                                        Ajlm_SOs_colocated = lambdaColocated(j,ll,mm)*(Xtildepilot(:,mm)'*DdeltaB_SOs)/Omegaj_SOs;
                                        hhatjlm_SOs_colocated = Yj_SOs_colocated*transpose(Ajlm_SOs_colocated);
                                        
                                        %Compute one part of the matrix
                                        %G_{jlm} and add it to Z_j
                                        outerProduct_SOs_colocated = rho*(hhatjlm_SOs_colocated*hhatjlm_SOs_colocated');
                                        Zj_SOs_colocated = Zj_SOs_colocated + outerProduct_SOs_colocated + diag(diag(outerProduct_SOs_colocated))*kappa2 + ( rho*lambdaColocated(j,ll,mm)*(1-Ajlm_SOs_colocated*DdeltaB_SOs*Xtildepilot(:,mm))*(1+kappa2)) *eye(N);
                                        
                                        %Compute the channel estimate
                                        %\hat{h}_{jlm} for a CO using Theorem 1
                                        Ajlm_CO_colocated = lambdaColocated(j,ll,mm)*(Xtildepilot(:,mm)'*DdeltaB_CO)/Omegaj_CO;
                                        hhatjlm_CO_colocated = Yj_CO_colocated*transpose(Ajlm_CO_colocated);
                                        
                                        %Compute one part of the matrix
                                        %G_{jlm} and add it to Z_j
                                        outerProduct_CO_colocated = rho*(hhatjlm_CO_colocated*hhatjlm_CO_colocated');
                                        Zj_CO_colocated = Zj_CO_colocated + outerProduct_CO_colocated + diag(diag(outerProduct_CO_colocated))*kappa2 + ( rho*lambdaColocated(j,ll,mm)*(1-Ajlm_CO_colocated*DdeltaB_CO*Xtildepilot(:,mm))*(1+kappa2)) *eye(N);
                                        
                                    end
                                    
                                    %Consider distributed deployments
                                    if runDistributedDeployment == true
                                        
                                        %Compute the channel estimate
                                        %\hat{h}_{jlm} for SOs using Theorem 1
                                        Ajlm_SOs_distributed = kron(Xtildepilot(:,mm)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,ll,mm)))/Psi_tildej_SOs_distributed;
                                        hhatjlm_SOs_distributed = reshape(reshape(Yj_SOs_distributed,[N/A A*Ksite])*transpose(Ajlm_SOs_distributed),[N 1]);  %kron(Ajlm_SOs_distributed,eye(N/A))*Yj_SOs_distributed(:);
                                        
                                        %Compute one part of the matrix
                                        %G_{jlm} and add it to Z_j
                                        lambdaVector = kron(lambdaDistributed(:,j,ll,mm),ones(N/A,1));
                                        outerProduct_SOs_distributed = rho*(hhatjlm_SOs_distributed*hhatjlm_SOs_distributed');
                                        Zj_SOs_distributed = Zj_SOs_distributed + outerProduct_SOs_distributed + diag(diag(outerProduct_SOs_distributed))*kappa2 + rho*(1+kappa2)*(diag(lambdaVector) - kron((kron(Xtildepilot(:,mm)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,ll,mm)))/Psi_tildej_SOs_distributed)*(kron(Xtildepilot(:,mm)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,ll,mm)))'),eye(N/A)));
                                        
                                        %Compute the channel estimate
                                        %\hat{h}_{jlm} for a CO using Theorem 1
                                        Ajlm_CO_distributed = kron(Xtildepilot(:,mm)'*DdeltaB_CO,diag(lambdaDistributed(:,j,ll,mm)))/Psi_tildej_CO_distributed;
                                        hhatjlm_CO_distributed = reshape(reshape(Yj_CO_distributed,[N/A A*Ksite])*transpose(Ajlm_CO_distributed),[N 1]); %kron(Ajlm_CO_distributed,eye(N/A))*Yj_CO_distributed(:);
                                        
                                        %Compute one part of the matrix
                                        %G_{jlm} and add it to Z_j
                                        outerProduct_CO_distributed = rho*(hhatjlm_CO_distributed*hhatjlm_CO_distributed');
                                        Zj_CO_distributed = Zj_CO_distributed + outerProduct_CO_distributed + diag(diag(outerProduct_CO_distributed))*kappa2 + rho*(1+kappa2)*(diag(lambdaVector) - kron((kron(Xtildepilot(:,mm)'*DdeltaB_CO,diag(lambdaDistributed(:,j,ll,mm)))/Psi_tildej_CO_distributed)*(kron(Xtildepilot(:,mm)'*DdeltaB_CO,diag(lambdaDistributed(:,j,ll,mm)))'),eye(N/A)));
                                        
                                    end
                                    
                                end
                                
                            end
                            
                        end
                        
                    end
                    
                    
                    %Go through all UEs in the middle cell
                    for k = 1:Ksite
                        
                        if runMonteCarloSimulations == true || r == 1
                            
                            %Compute the non-identity-matrix part of the scaling
                            %factor in the LMMSE estimate in Eq. (16) for
                            %co-located deployments
                            if runColocatedDeployment == true
                                Ajjk_CO_colocated = lambdaColocated(j,j,k)*(Xtildepilot(:,k)'*DdeltaB_CO)/Omegaj_CO;
                                Ajjk_SOs_colocated = lambdaColocated(j,j,k)*(Xtildepilot(:,k)'*DdeltaB_SOs)/Omegaj_SOs;
                            end
                            
                            %Compute the matrix in Eq. (37) for distributed
                            %deployments. Notice that this is the matrix
                            %that defines the LMMSE estimate in Eq. (9).
                            %We also precompute terms of the type in Eq.
                            %(27), namely \tilde{Psi}_j^{-1} ( D_{delta(t)} \tilde{x}_kl \kronecker e_{a_2}
                            if runDistributedDeployment == true
                                Ajjk_CO_distributed = kron(Xtildepilot(:,k)'*DdeltaB_CO,diag(lambdaDistributed(:,j,j,k)))/Psi_tildej_CO_distributed;
                                Ajjk_SOs_distributed = kron(Xtildepilot(:,k)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,j,k)))/Psi_tildej_SOs_distributed;
                                
                                %Precompute terms of the type
                                %\tilde{Psi}_j^{-1} ( D_{delta(t)} \tilde{x}_kl \kronecker e_{a_2}
                                ejjk_CO_distributed = zeros(B*A,A);
                                ejjk_SOs_distributed = zeros(B*A,A);
                                for a = 1:A
                                    ejjk_CO_distributed(:,a) = Psi_tildej_CO_distributed\kron(DdeltaB_CO*Xtildepilot(:,k),I_A(:,a));
                                    ejjk_SOs_distributed(:,a) = Psi_tildej_SOs_distributed\kron(DdeltaB_SOs*Xtildepilot(:,k),I_A(:,a));
                                end
                            end
                            
                        end
                        
                        
                        %Compute the MRC filter and the MMSE filters, defined in Eq. (33)
                        if runMonteCarloSimulations == true
                            
                            if runColocatedDeployment == true
                                
                                MRCfilter_SOs_colocated = Yj_SOs_colocated*transpose(Ajjk_SOs_colocated);
                                MRCfilter_CO_colocated = Yj_CO_colocated*transpose(Ajjk_CO_colocated);
                                
                                if runMonteCarlo_MMSE == true
                                    MMSEfilter_SOs_colocated = Zj_SOs_colocated\MRCfilter_SOs_colocated;
                                    MMSEfilter_CO_colocated = Zj_CO_colocated\MRCfilter_CO_colocated;
                                end
                                
                            end
                            
                            if runDistributedDeployment == true
                                
                                MRCfilter_SOs_distributed = kron(Ajjk_SOs_distributed,eye(N/A))*Yj_SOs_distributed(:);
                                MRCfilter_CO_distributed = kron(Ajjk_CO_distributed,eye(N/A))*Yj_CO_distributed(:);
                                
                                if runMonteCarlo_MMSE == true
                                    MMSEfilter_SOs_distributed = Zj_SOs_distributed\MRCfilter_SOs_distributed;
                                    MMSEfilter_CO_distributed = Zj_CO_distributed\MRCfilter_CO_distributed;
                                end
                                
                            end
                            
                        end
                        
                        
                        %Compute the expectations in Lemma 1 for MRC
                        %filtering, using expressions in Theorem 2 and
                        %Corollary 2.
                        if r == 1
                            
                            if runColocatedDeployment == true
                                
                                %Precompute some terms
                                Ajjk_CO_colocatedD = Ajjk_CO_colocated * DdeltaB_CO;
                                Ajjk_SOs_colocatedD = Ajjk_SOs_colocated * DdeltaB_SOs;
                                
                                %Compute the expectations in Eqs. (20) and
                                %(21), but using the simplified expressions
                                %in Corollary 2.
                                filterNorm_MRC_CO_colocated_analytical(j,k,:) = (N*lambdaColocated(j,j,k)*Ajjk_CO_colocatedD*Xtildepilot(:,k)) * reshape(deltaTB2_CO,[1 1 T-B]);
                                filterNorm_MRC_SOs_colocated_analytical(j,k,:) = (N*lambdaColocated(j,j,k)*Ajjk_SOs_colocatedD*Xtildepilot(:,k)) * reshape(deltaTB2_SOs,[1 1 T-B]);
                                firstMoment_MRC_CO_colocated_analytical(j,k,:) = filterNorm_MRC_CO_colocated_analytical(j,k,:);
                                firstMoment_MRC_SOs_colocated_analytical(j,k,:) = filterNorm_MRC_SOs_colocated_analytical(j,k,:);
                                
                                %Initialize the computation of the
                                %expectation in Eq. (23), but using the
                                %simplified expressions in Corollary 2.
                                distortionTerm_MRC_CO_colocated_analytical(j,k,:) = kappa2*sum(sum(lambdaColocated(j,:,:)))*rho*filterNorm_MRC_CO_colocated_analytical(j,k,:);
                                distortionTerm_MRC_SOs_colocated_analytical(j,k,:) = kappa2*sum(sum(lambdaColocated(j,:,:)))*rho*filterNorm_MRC_SOs_colocated_analytical(j,k,:);
                                
                                %Compute the asymptotic signal term from
                                %Eq. (26) and prepare computation of the
                                %interference term from Eqs. (27) and (28).
                                if nIndex == 1
                                    asymptoticSignal_MRC_CO_colocated = rho*lambdaColocated(j,j,k)^2*((Ajjk_CO_colocatedD*Xtildepilot(:,k)) * deltaTB2_CO).^2;
                                    asymptoticSignal_MRC_SOs_colocated = rho*lambdaColocated(j,j,k)^2*((Ajjk_SOs_colocatedD*Xtildepilot(:,k)) * deltaTB2_SOs).^2;
                                    asymptoticInterference_MRC_SOs_colocated = - asymptoticSignal_MRC_SOs_colocated;
                                    asymptoticInterference_MRC_CO_colocated = - asymptoticSignal_MRC_CO_colocated;
                                end
                                
                                
                                %Go through all cells and UEs, in order to
                                %compute the sum of the interference terms
                                %and the distortion noise. Note that we
                                %compute the results for different t-values
                                %at the same time, by multiplying with
                                %deltaTB which determines how the MRC
                                %filter changes over time.
                                for ll = 1:BSs
                                    for mm = 1:Ksite
                                        
                                        %Precompute some multiplications in
                                        %Corollary 2.
                                        bXb_CO = Ajjk_CO_colocated *(XlmCO(:,:,mm) *Ajjk_CO_colocated');
                                        bXb2_CO = Ajjk_CO_colocated *(XlmbarCO(:,:,mm) *Ajjk_CO_colocated');
                                        bXb_SOs = Ajjk_SOs_colocated *(XlmSOs(:,:,mm) *Ajjk_SOs_colocated');
                                        bXb2_SOs = Ajjk_SOs_colocated *(XlmbarSOs(:,:,mm) *Ajjk_SOs_colocated');
                                        
                                        %Compute the sum of the second
                                        %order moments in Eq. (22), but
                                        %using the expressions in Corollary 2.
                                        secondMoments_MRC_SOs_colocated_analytical(j,k,:) = secondMoments_MRC_SOs_colocated_analytical(j,k,:) + (lambdaColocated(j,ll,mm)*N*lambdaColocated(j,j,k)*Ajjk_SOs_colocatedD*Xtildepilot(:,k)) * reshape(deltaTB2_SOs,[1 1 T-B]) + (N*lambdaColocated(j,ll,mm)^2 * bXb_SOs) * reshape(deltaTB2_SOs,[1 1 T-B]) + (N*(N-1)*lambdaColocated(j,ll,mm)^2 * abs(Ajjk_SOs_colocatedD*Xtildepilot(:,mm))^2) * reshape(deltaTB2_SOs.^2,[1 1 T-B]);
                                        secondMoments_MRC_CO_colocated_analytical(j,k,:) = secondMoments_MRC_CO_colocated_analytical(j,k,:) + (lambdaColocated(j,ll,mm)*N*lambdaColocated(j,j,k)*Ajjk_CO_colocatedD*Xtildepilot(:,k)) * reshape(deltaTB2_CO,[1 1 T-B]) + (N*lambdaColocated(j,ll,mm)^2 * bXb_CO + N*(N-1)*lambdaColocated(j,ll,mm)^2 * bXb2_CO) * reshape(deltaTB2_CO,[1 1 T-B]);
                                        
                                        %Compute and sum up the expectation
                                        %in Eq. (23), but using the
                                        %expression in Corollary 2.
                                        distortionTerm_MRC_CO_colocated_analytical(j,k,:) = distortionTerm_MRC_CO_colocated_analytical(j,k,:) + kappa2*rho*N*lambdaColocated(j,ll,mm)^2 * bXb_CO * reshape(deltaTB2_CO,[1 1 T-B]);
                                        distortionTerm_MRC_SOs_colocated_analytical(j,k,:) = distortionTerm_MRC_SOs_colocated_analytical(j,k,:) + kappa2*rho*N*lambdaColocated(j,ll,mm)^2 * bXb_SOs * reshape(deltaTB2_SOs,[1 1 T-B]);
                                        
                                        %Sum up the asymptotic interference
                                        %terms as in Eqs. (27) and (28).
                                        if nIndex == 1
                                            asymptoticInterference_MRC_SOs_colocated = asymptoticInterference_MRC_SOs_colocated + rho*lambdaColocated(j,ll,mm)^2 * abs(Ajjk_SOs_colocated * DdeltaB_SOs*Xtildepilot(:,mm))^2 * deltaTB2_SOs.^2;
                                            asymptoticInterference_MRC_CO_colocated = asymptoticInterference_MRC_CO_colocated + rho*lambdaColocated(j,ll,mm)^2 * bXb2_CO * deltaTB2_CO;
                                        end
                                        
                                    end
                                end
                                
                                %Compute the asymptotic rates with MRC,
                                %using Eq. (25) when N->infinity.
                                if nIndex == 1
                                    rates_MRC_SOs_colocated_asymptotics(j,k,s,ex) = sum(log2(1+real(asymptoticSignal_MRC_SOs_colocated)./real(asymptoticInterference_MRC_SOs_colocated)))/T;
                                    rates_MRC_CO_colocated_asymptotics(j,k,s,ex) = sum(log2(1+real(asymptoticSignal_MRC_CO_colocated)./real(asymptoticInterference_MRC_CO_colocated)))/T;
                                end
                                
                            end
                            
                            
                            if runDistributedDeployment == true
                                
                                %Compute the expectations in Eqs. (20) and (21).
                                filterNorm_MRC_SOs_distributed_analytical(j,k,:) = (N/A)*trace(Ajjk_SOs_distributed* kron(Xtildepilot(:,k)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_SOs,[1 1 T-B]);
                                filterNorm_MRC_CO_distributed_analytical(j,k,:) = (N/A)*trace(Ajjk_CO_distributed* kron(Xtildepilot(:,k)'*DdeltaB_CO,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_CO,[1 1 T-B]);
                                firstMoment_MRC_SOs_distributed_analytical(j,k,:) = filterNorm_MRC_SOs_distributed_analytical(j,k,:);
                                firstMoment_MRC_CO_distributed_analytical(j,k,:) = filterNorm_MRC_CO_distributed_analytical(j,k,:);
                                
                                %Initialize the computation of the
                                %expectations in Eqs. (22) and (23).
                                sumLambdaAjlm = sum(sum(lambdaDistributed(:,j,:,:),3),4);
                                traceFormulaA_SOs =  (N/A)*trace(diag(sumLambdaAjlm)*Ajjk_SOs_distributed* kron(Xtildepilot(:,k)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_SOs,[1 1 T-B]);
                                traceFormulaA_CO =  (N/A)*trace(diag(sumLambdaAjlm)*Ajjk_CO_distributed* kron(Xtildepilot(:,k)'*DdeltaB_CO,diag(lambdaDistributed(:,j,j,k)))') * reshape(deltaTB2_CO,[1 1 T-B]);
                                distortionTerm_MRC_SOs_distributed_analytical(j,k,:) = kappa2*rho*traceFormulaA_SOs;
                                distortionTerm_MRC_CO_distributed_analytical(j,k,:) = kappa2*rho*traceFormulaA_CO;
                                secondMoments_MRC_SOs_distributed_analytical(j,k,:) = traceFormulaA_SOs;
                                secondMoments_MRC_CO_distributed_analytical(j,k,:) = traceFormulaA_CO;
                                
                                %Compute the asymptotic signal term from
                                %Eq. (26) and prepare computation of the
                                %interference term from Eqs. (27) and (28).
                                if nIndex == 1
                                    asymptoticSignal_MRC_SOs_distributed = rho*(trace(Ajjk_SOs_distributed* kron(Xtildepilot(:,k)'*DdeltaB_SOs,diag(lambdaDistributed(:,j,j,k)))') * deltaTB2_SOs).^2;
                                    asymptoticSignal_MRC_CO_distributed = rho*(trace(Ajjk_CO_distributed* kron(Xtildepilot(:,k)'*DdeltaB_CO,diag(lambdaDistributed(:,j,j,k)))') * deltaTB2_CO).^2;
                                    asymptoticInterference_MRC_SOs_distributed = - asymptoticSignal_MRC_SOs_distributed;
                                    asymptoticInterference_MRC_CO_distributed = - asymptoticSignal_MRC_CO_distributed;
                                end
                                
                                %Go through all cells and UEs, in order to
                                %compute the sum of the interference terms
                                %and the distortion noise. Note that we
                                %compute the results for different t-values
                                %at the same time, by multiplying with
                                %deltaTB which determines how the MRC
                                %filter changes over time.
                                for ll = 1:BSs
                                    for mm = 1:Ksite
                                        
                                        %Compute the sum of the second
                                        %order moments in Eq. (22) and the
                                        %compute and sum up the expectation
                                        %in Eq. (23).
                                        secondMoments_MRC_SOs_distributed_analytical(j,k,:) = secondMoments_MRC_SOs_distributed_analytical(j,k,:) + (N/A)^2*trace(Ajjk_SOs_distributed* kron(Xtildepilot(:,mm)'*DdeltaB_SOs, diag(lambdaDistributed(:,j,ll,mm)))')^2 * reshape(deltaTB2_SOs.^2,[1 1 T-B]);
                                        
                                        for a1 = 1:A
                                            
                                            secondMoments_MRC_SOs_distributed_analytical(j,k,:) = secondMoments_MRC_SOs_distributed_analytical(j,k,:) + (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_SOs_distributed(:,a1)'*( kron(XlmSOs(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_SOs_distributed(:,a1) * reshape(deltaTB2_SOs,[1 1 T-B]) - (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_SOs_distributed(:,a1)'*( kron( DdeltaB_SOs*Xtildepilot(:,mm)*Xtildepilot(:,mm)'*DdeltaB_SOs ,I_A(:,a1)*I_A(:,a1)') )*ejjk_SOs_distributed(:,a1) * reshape(deltaTB2_SOs.^2,[1 1 T-B]);
                                            secondMoments_MRC_CO_distributed_analytical(j,k,:) = secondMoments_MRC_CO_distributed_analytical(j,k,:) + (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_CO_distributed(:,a1)'*( kron(XlmCO(:,:,mm)-XlmbarCO(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_CO_distributed(:,a1) * reshape(deltaTB2_CO,[1 1 T-B]);
                                            
                                            distortionTerm_MRC_SOs_distributed_analytical(j,k,:) = distortionTerm_MRC_SOs_distributed_analytical(j,k,:) +  kappa2*rho* (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_SOs_distributed(:,a1)'*( kron(XlmSOs(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_SOs_distributed(:,a1) * reshape(deltaTB2_SOs,[1 1 T-B]);
                                            distortionTerm_MRC_CO_distributed_analytical(j,k,:) = distortionTerm_MRC_CO_distributed_analytical(j,k,:) +  kappa2*rho* (N/A)*lambdaDistributed(a1,j,j,k)^2*lambdaDistributed(a1,j,ll,mm)^2* ejjk_CO_distributed(:,a1)'*( kron(XlmCO(:,:,mm) ,I_A(:,a1)*I_A(:,a1)') )*ejjk_CO_distributed(:,a1) * reshape(deltaTB2_CO,[1 1 T-B]);
                                            
                                            for a2 = 1:A
                                                
                                                secondMoments_MRC_CO_distributed_analytical(j,k,:) = secondMoments_MRC_CO_distributed_analytical(j,k,:) + (N/A)^2*lambdaDistributed(a1,j,j,k)*lambdaDistributed(a1,j,ll,mm)*lambdaDistributed(a2,j,j,k)*lambdaDistributed(a2,j,ll,mm)* ejjk_CO_distributed(:,a1)'*( kron(XlmbarCO(:,:,mm) ,I_A(:,a1)*I_A(:,a2)') )*ejjk_CO_distributed(:,a2) * reshape(deltaTB2_CO,[1 1 T-B]);
                                                
                                                %Sum up the asymptotic interference
                                                %terms as in Eqs. (27) and (28).
                                                if nIndex==1
                                                    asymptoticInterference_MRC_CO_distributed = asymptoticInterference_MRC_CO_distributed + rho*lambdaDistributed(a1,j,j,k)*lambdaDistributed(a1,j,ll,mm)*lambdaDistributed(a2,j,j,k)*lambdaDistributed(a2,j,ll,mm)* ejjk_CO_distributed(:,a1)'*( kron(XlmbarCO(:,:,mm) ,I_A(:,a1)*I_A(:,a2)') )*ejjk_CO_distributed(:,a2) * deltaTB2_CO;
                                                end
                                                
                                            end
                                            
                                        end
                                        
                                        %Sum up the asymptotic interference
                                        %terms as in Eqs. (27) and (28).
                                        if nIndex==1
                                            asymptoticInterference_MRC_SOs_distributed = asymptoticInterference_MRC_SOs_distributed + rho*trace(Ajjk_SOs_distributed* kron(Xtildepilot(:,mm)'*DdeltaB_SOs, diag(lambdaDistributed(:,j,ll,mm)))')^2 * deltaTB2_SOs.^2;
                                        end
                                        
                                    end
                                    
                                end
                                
                                
                                %Compute the asymptotic rates with MRC,
                                %using Eq. (25) when N->infinity.
                                if nIndex == 1
                                    rates_MRC_SOs_distributed_asymptotics(j,k,s,ex) = sum(log2(1+real(asymptoticSignal_MRC_SOs_distributed)./real(asymptoticInterference_MRC_SOs_distributed)))/T;
                                    rates_MRC_CO_distributed_asymptotics(j,k,s,ex) = sum(log2(1+real(asymptoticSignal_MRC_CO_distributed)./real(asymptoticInterference_MRC_CO_distributed)))/T;
                                end
                                
                            end
                            
                        end
                        
                        
                        %Compute the expectations in Lemma 1 by Monte Carlo
                        %simulations, for MRC and MMSE receive filters
                        if runMonteCarloSimulations == true
                            
                            phijCO_N = repmat(phijCO(1,B+1:T,r),[N 1]);
                            
                            if runMonteCarlo_MRC == true
                                
                                if runColocatedDeployment == true
                                    
                                    filterNorm_MRC_SOs_colocated_numerical(j,k,:)= filterNorm_MRC_SOs_colocated_numerical(j,k,:) + norm(MRCfilter_SOs_colocated)^2*reshape(deltaTB2_SOs,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MRC_SOs_colocated_numerical(j,k,:) = firstMoment_MRC_SOs_colocated_numerical(j,k,:) + reshape(((MRCfilter_SOs_colocated*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*Hcolocated(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MRC_SOs_colocated_numerical(j,k,:) = distortionTerm_MRC_SOs_colocated_numerical(j,k,:) + reshape(abs(sum(conj(MRCfilter_SOs_colocated*deltaTB_SOs) .* upsilon_colocatedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MRC_SOs_colocated_numerical(j,k,:) = secondMoments_MRC_SOs_colocated_numerical(j,k,:) + reshape(sum(abs(((MRCfilter_SOs_colocated*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*reshape(Hcolocated(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                    filterNorm_MRC_CO_colocated_numerical(j,k,:)= filterNorm_MRC_CO_colocated_numerical(j,k,:) + norm(MRCfilter_CO_colocated)^2*reshape(deltaTB2_CO,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MRC_CO_colocated_numerical(j,k,:) = firstMoment_MRC_CO_colocated_numerical(j,k,:) + reshape(((MRCfilter_CO_colocated*deltaTB_CO).*exp(-1i*phijCO_N))'*Hcolocated(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MRC_CO_colocated_numerical(j,k,:) = distortionTerm_MRC_CO_colocated_numerical(j,k,:) + reshape(abs(sum(conj(MRCfilter_CO_colocated*deltaTB_CO) .* upsilon_colocatedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MRC_CO_colocated_numerical(j,k,:) = secondMoments_MRC_CO_colocated_numerical(j,k,:) + reshape(sum(abs(((MRCfilter_CO_colocated*deltaTB_CO).*exp(-1i*phijCO_N))'*reshape(Hcolocated(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                end
                                
                                if runDistributedDeployment == true
                                    
                                    filterNorm_MRC_SOs_distributed_numerical(j,k,:)= filterNorm_MRC_SOs_distributed_numerical(j,k,:) + norm(MRCfilter_SOs_distributed)^2*reshape(deltaTB2_SOs,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MRC_SOs_distributed_numerical(j,k,:) = firstMoment_MRC_SOs_distributed_numerical(j,k,:) + reshape(((MRCfilter_SOs_distributed*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*Hdistributed(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MRC_SOs_distributed_numerical(j,k,:) = distortionTerm_MRC_SOs_distributed_numerical(j,k,:) + reshape(abs(sum(conj(MRCfilter_SOs_distributed*deltaTB_SOs) .* upsilon_distributedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MRC_SOs_distributed_numerical(j,k,:) = secondMoments_MRC_SOs_distributed_numerical(j,k,:) + reshape(sum(abs(((MRCfilter_SOs_distributed*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*reshape(Hdistributed(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                    filterNorm_MRC_CO_distributed_numerical(j,k,:)= filterNorm_MRC_CO_distributed_numerical(j,k,:) + norm(MRCfilter_CO_distributed)^2*reshape(deltaTB2_CO,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MRC_CO_distributed_numerical(j,k,:) = firstMoment_MRC_CO_distributed_numerical(j,k,:) + reshape(((MRCfilter_CO_distributed*deltaTB_CO).*exp(-1i*phijCO_N))'*Hdistributed(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MRC_CO_distributed_numerical(j,k,:) = distortionTerm_MRC_CO_distributed_numerical(j,k,:) + reshape(abs(sum(conj(MRCfilter_CO_distributed*deltaTB_CO) .* upsilon_distributedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MRC_CO_distributed_numerical(j,k,:) = secondMoments_MRC_CO_distributed_numerical(j,k,:) + reshape(sum(abs(((MRCfilter_CO_distributed*deltaTB_CO).*exp(-1i*phijCO_N))'*reshape(Hdistributed(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                end
                                
                            end
                            
                            
                            
                            if runMonteCarlo_MRC_upper == true && scalingExponents(ex)==-1
                                
                                if runColocatedDeployment == true
                                    
                                    rates_MRC_upper_colocated_numerical(j,k,s,nIndex,ex) = rates_MRC_upper_colocated_numerical(j,k,s,nIndex,ex) + (1-B/T)*log2(1+rho*norm(MRCfilter_SOs_colocated).^4 / ( abs(MRCfilter_SOs_colocated'*(Zj_SOs_colocated-xi*eye(N))*MRCfilter_SOs_colocated) - rho*norm(MRCfilter_SOs_colocated).^4 + xi*norm(MRCfilter_SOs_colocated).^2) )/nbrOfMonteCarloRealizations;
                                    
                                end
                                
                                if runDistributedDeployment == true
                                    
                                    rates_MRC_upper_distributed_numerical(j,k,s,nIndex,ex) = rates_MRC_upper_distributed_numerical(j,k,s,nIndex,ex) + (1-B/T)*log2(1+rho*norm(MRCfilter_SOs_distributed).^4 / ( abs(MRCfilter_SOs_distributed'*(Zj_SOs_distributed-xi*eye(N))*MRCfilter_SOs_distributed) - rho*norm(MRCfilter_SOs_distributed).^4 + xi*norm(MRCfilter_SOs_distributed).^2) )/nbrOfMonteCarloRealizations;
                                    
                                end
                                
                            end
                            
                            
                            if runMonteCarlo_MMSE == true
                                
                                if runColocatedDeployment == true
                                    
                                    filterNorm_MMSE_SOs_colocated_numerical(j,k,:)= filterNorm_MMSE_SOs_colocated_numerical(j,k,:) + norm(MMSEfilter_SOs_colocated)^2*reshape(deltaTB2_SOs,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MMSE_SOs_colocated_numerical(j,k,:) = firstMoment_MMSE_SOs_colocated_numerical(j,k,:) + reshape(((MMSEfilter_SOs_colocated*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*Hcolocated(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MMSE_SOs_colocated_numerical(j,k,:) = distortionTerm_MMSE_SOs_colocated_numerical(j,k,:) + reshape(abs(sum(conj(MMSEfilter_SOs_colocated*deltaTB_SOs) .* upsilon_colocatedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MMSE_SOs_colocated_numerical(j,k,:) = secondMoments_MMSE_SOs_colocated_numerical(j,k,:) + reshape(sum(abs(((MMSEfilter_SOs_colocated*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*reshape(Hcolocated(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                    filterNorm_MMSE_CO_colocated_numerical(j,k,:)= filterNorm_MMSE_CO_colocated_numerical(j,k,:) + norm(MMSEfilter_CO_colocated)^2*reshape(deltaTB2_CO,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MMSE_CO_colocated_numerical(j,k,:) = firstMoment_MMSE_CO_colocated_numerical(j,k,:) + reshape(((MMSEfilter_CO_colocated*deltaTB_CO).*exp(-1i*phijCO_N))'*Hcolocated(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MMSE_CO_colocated_numerical(j,k,:) = distortionTerm_MMSE_CO_colocated_numerical(j,k,:) + reshape(abs(sum(conj(MMSEfilter_CO_colocated*deltaTB_CO) .* upsilon_colocatedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MMSE_CO_colocated_numerical(j,k,:) = secondMoments_MMSE_CO_colocated_numerical(j,k,:) + reshape(sum(abs(((MMSEfilter_CO_colocated*deltaTB_CO).*exp(-1i*phijCO_N))'*reshape(Hcolocated(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                end
                                
                                if runDistributedDeployment == true
                                    
                                    filterNorm_MMSE_SOs_distributed_numerical(j,k,:)= filterNorm_MMSE_SOs_distributed_numerical(j,k,:) + norm(MMSEfilter_SOs_distributed)^2*reshape(deltaTB2_SOs,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MMSE_SOs_distributed_numerical(j,k,:) = firstMoment_MMSE_SOs_distributed_numerical(j,k,:) + reshape(((MMSEfilter_SOs_distributed*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*Hdistributed(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MMSE_SOs_distributed_numerical(j,k,:) = distortionTerm_MMSE_SOs_distributed_numerical(j,k,:) + reshape(abs(sum(conj(MMSEfilter_SOs_distributed*deltaTB_SOs) .* upsilon_distributedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MMSE_SOs_distributed_numerical(j,k,:) = secondMoments_MMSE_SOs_distributed_numerical(j,k,:) + reshape(sum(abs(((MMSEfilter_SOs_distributed*deltaTB_SOs).*exp(-1i*phijSOs(:,B+1:T,r)))'*reshape(Hdistributed(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                    filterNorm_MMSE_CO_distributed_numerical(j,k,:)= filterNorm_MMSE_CO_distributed_numerical(j,k,:) + norm(MMSEfilter_CO_distributed)^2*reshape(deltaTB2_CO,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    firstMoment_MMSE_CO_distributed_numerical(j,k,:) = firstMoment_MMSE_CO_distributed_numerical(j,k,:) + reshape(((MMSEfilter_CO_distributed*deltaTB_CO).*exp(-1i*phijCO_N))'*Hdistributed(activeAntennas,j,j,k,r),[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    distortionTerm_MMSE_CO_distributed_numerical(j,k,:) = distortionTerm_MMSE_CO_distributed_numerical(j,k,:) + reshape(abs(sum(conj(MMSEfilter_CO_distributed*deltaTB_CO) .* upsilon_distributedj(:,B+1:T,r),1)).^2,[1 1 T-B])/nbrOfMonteCarloRealizations;
                                    secondMoments_MMSE_CO_distributed_numerical(j,k,:) = secondMoments_MMSE_CO_distributed_numerical(j,k,:) + reshape(sum(abs(((MMSEfilter_CO_distributed*deltaTB_CO).*exp(-1i*phijCO_N))'*reshape(Hdistributed(activeAntennas,j,:,:,r),[N BSs*Ksite])).^2,2)/nbrOfMonteCarloRealizations,[1 1 T-B]);
                                    
                                end
                                
                            end
                            
                        end
                        
                    end
                    
                end
                
            end
            
            
            %Compute the rates by using the analytic expressions in Theorem
            %1 or Monte-Carlo simualations for each of the expectations in
            %Lemma 1. This is for the co-located deployment.
            if runColocatedDeployment == true
                
                rates_MRC_SOs_colocated_analytical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_SOs_colocated_analytical).^2 ./ ( rho* real(secondMoments_MRC_SOs_colocated_analytical )- rho*abs(firstMoment_MRC_SOs_colocated_analytical).^2 + real(distortionTerm_MRC_SOs_colocated_analytical) + xi*real(filterNorm_MRC_SOs_colocated_analytical))),3)/T;
                rates_MRC_CO_colocated_analytical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_CO_colocated_analytical).^2 ./ ( rho* real(secondMoments_MRC_CO_colocated_analytical) - rho*abs(firstMoment_MRC_CO_colocated_analytical).^2 + real(distortionTerm_MRC_CO_colocated_analytical) + xi*real(filterNorm_MRC_CO_colocated_analytical))),3)/T;
                
                if runMonteCarlo_MRC == true
                    rates_MRC_SOs_colocated_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_SOs_colocated_numerical).^2 ./ ( rho* secondMoments_MRC_SOs_colocated_numerical - rho*abs(firstMoment_MRC_SOs_colocated_numerical).^2 + distortionTerm_MRC_SOs_colocated_numerical + xi*filterNorm_MRC_SOs_colocated_numerical)),3)/T;
                    rates_MRC_CO_colocated_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_CO_colocated_numerical).^2 ./ ( rho* secondMoments_MRC_CO_colocated_numerical - rho*abs(firstMoment_MRC_CO_colocated_numerical).^2 + distortionTerm_MRC_CO_colocated_numerical + xi*filterNorm_MRC_CO_colocated_numerical)),3)/T;
                end
                
                if runMonteCarlo_MMSE == true
                    rates_MMSE_SOs_colocated_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MMSE_SOs_colocated_numerical).^2 ./ ( rho* secondMoments_MMSE_SOs_colocated_numerical - rho*abs(firstMoment_MMSE_SOs_colocated_numerical).^2 + distortionTerm_MMSE_SOs_colocated_numerical + xi*filterNorm_MMSE_SOs_colocated_numerical)),3)/T;
                    rates_MMSE_CO_colocated_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MMSE_CO_colocated_numerical).^2 ./ ( rho* secondMoments_MMSE_CO_colocated_numerical - rho*abs(firstMoment_MMSE_CO_colocated_numerical).^2 + distortionTerm_MMSE_CO_colocated_numerical + xi*filterNorm_MMSE_CO_colocated_numerical)),3)/T;
                end
                
            end
            
            
            %Compute the rates by using the analytic expressions in Theorem
            %1 or Monte-Carlo simualations for each of the expectations in
            %Lemma 1. This is for the distributed deployment.
            if runDistributedDeployment == true
                
                rates_MRC_SOs_distributed_analytical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_SOs_distributed_analytical).^2 ./ ( rho* real(secondMoments_MRC_SOs_distributed_analytical )- rho*abs(firstMoment_MRC_SOs_distributed_analytical).^2 + real(distortionTerm_MRC_SOs_distributed_analytical) + xi*real(filterNorm_MRC_SOs_distributed_analytical))),3)/T;
                rates_MRC_CO_distributed_analytical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_CO_distributed_analytical).^2 ./ ( rho* real(secondMoments_MRC_CO_distributed_analytical) - rho*abs(firstMoment_MRC_CO_distributed_analytical).^2 + real(distortionTerm_MRC_CO_distributed_analytical) + xi*real(filterNorm_MRC_CO_distributed_analytical))),3)/T;
                
                if runMonteCarlo_MRC == true
                    rates_MRC_SOs_distributed_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_SOs_distributed_numerical).^2 ./ ( rho* secondMoments_MRC_SOs_distributed_numerical - rho*abs(firstMoment_MRC_SOs_distributed_numerical).^2 + distortionTerm_MRC_SOs_distributed_numerical + xi*filterNorm_MRC_SOs_distributed_numerical)),3)/T;
                    rates_MRC_CO_distributed_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MRC_CO_distributed_numerical).^2 ./ ( rho* secondMoments_MRC_CO_distributed_numerical - rho*abs(firstMoment_MRC_CO_distributed_numerical).^2 + distortionTerm_MRC_CO_distributed_numerical + xi*filterNorm_MRC_CO_distributed_numerical)),3)/T;
                end
                
                if runMonteCarlo_MMSE == true
                    rates_MMSE_SOs_distributed_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MMSE_SOs_distributed_numerical).^2 ./ ( rho* secondMoments_MMSE_SOs_distributed_numerical - rho*abs(firstMoment_MMSE_SOs_distributed_numerical).^2 + distortionTerm_MMSE_SOs_distributed_numerical + xi*filterNorm_MMSE_SOs_distributed_numerical)),3)/T;
                    rates_MMSE_CO_distributed_numerical(:,:,s,nIndex,ex) = sum(log2(1+rho*abs(firstMoment_MMSE_CO_distributed_numerical).^2 ./ ( rho* secondMoments_MMSE_CO_distributed_numerical - rho*abs(firstMoment_MMSE_CO_distributed_numerical).^2 + distortionTerm_MMSE_CO_distributed_numerical + xi*filterNorm_MMSE_CO_distributed_numerical)),3)/T;
                end
                
            end
            
            
        end
        
    end
    
end



if simulateFigure == 4
    
    %Plot Figure 4 from the paper
    figure(4); hold on; box on;
    
    if runColocatedDeployment == true
        
        if runMonteCarlo_MRC_upper == true
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_upper_colocated_numerical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'r:','LineWidth',1);
        end
        
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_colocated_analytical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'r','LineWidth',1);
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_colocated_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'k--','LineWidth',1);
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_colocated_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'b-.','LineWidth',1);
        
        if runMonteCarlo_MRC_upper == true
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_upper_colocated_numerical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'rs','LineWidth',1);
        end
        
        if runMonteCarlo_MRC == true
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_colocated_numerical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'rd','LineWidth',1);
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_colocated_numerical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'kd','LineWidth',1);
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_colocated_numerical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'bd','LineWidth',1);
        end
        
    end
    
    if runDistributedDeployment == true
        if runMonteCarlo_MRC_upper == true
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_upper_distributed_numerical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'r:','LineWidth',1);
        end
        
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'r','LineWidth',1);
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'k--','LineWidth',1);
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_distributed_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'b-.','LineWidth',1);
        
        if runMonteCarlo_MRC_upper == true
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_upper_distributed_numerical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'rs','LineWidth',1);
        end
        
        if runMonteCarlo_MRC == true
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_numerical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'ro','LineWidth',1);
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_numerical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'ko','LineWidth',1);
            plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_distributed_numerical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'bo','LineWidth',1);
        end
        
        
    end
    
    legend('Ideal Hardware, (39) in [12]','Ideal Hardware, Lemma 1','Non-Ideal Hardware: SLOs','Non-Ideal Hardware: CLO','Location','SouthEast');
    xlabel('Number of Receive Antennas per Cell');
    ylabel('Average Rate per UE [bit/channel use]');
    
elseif simulateFigure == 5
    
    %Plot Figure 5 from the paper
    figure(5); hold on; box on;
    
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'r','LineWidth',1);
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'k--','LineWidth',1);
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_distributed_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'b-.','LineWidth',1);
    
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,3),3),2),1),[length(Nantennas) 1])/(Ksite),'r','LineWidth',1);
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,4),3),2),1),[length(Nantennas) 1])/(Ksite),'k--','LineWidth',1);
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_distributed_analytical(:,:,:,:,4),3),2),1),[length(Nantennas) 1])/(Ksite),'b-.','LineWidth',1);
    
    for ex = 1:length(scalingExponents)
        plot(Nantennas,ones(1,length(Nantennas))*sum(sum(mean(rates_MRC_SOs_distributed_asymptotics(:,:,:,ex),3),2),1)/(Ksite),'k:','LineWidth',1);
        plot(Nantennas,ones(1,length(Nantennas))*sum(sum(mean(rates_MRC_CO_distributed_asymptotics(:,:,:,ex),3),2),1)/(Ksite),'k:','LineWidth',1);
    end
    
    legend('Ideal Hardware','Non-Ideal Hardware: SLOs','Non-Ideal Hardware: CLO','Location','SouthEast');
    xlabel('Number of Receive Antennas per Cell');
    ylabel('Average Rate per UE [bit/channel use]');
    set(gca,'XScale','Log');
    axis([0 Nantennas(end) 0 10]);
    
    
elseif simulateFigure == 7
    
    %Plot Figure 7 from the paper
    figure(7); hold on; box on;
    
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'r-','LineWidth',1);
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'k--','LineWidth',1);
    plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_distributed_analytical(:,:,:,:,2),3),2),1),[length(Nantennas) 1])/(Ksite),'b-.','LineWidth',1);
    
    for ex = 3:length(scalingExponents)
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,ex),3),2),1),[length(Nantennas) 1])/(Ksite),'k--','LineWidth',1);
        plot(Nantennas,reshape(sum(sum(mean(rates_MRC_CO_distributed_analytical(:,:,:,:,ex),3),2),1),[length(Nantennas) 1])/(Ksite),'b-.','LineWidth',1);
    end
    
    %Find the maximum on the curve when the scaling law is not satisfied.
    rates_SOs_scaling1 = reshape(sum(sum(mean(rates_MRC_SOs_distributed_analytical(:,:,:,:,4),3),2),1),[length(Nantennas) 1])/(Ksite);
    [maxRate,ind] = max(rates_SOs_scaling1);
    plot(Nantennas(ind),maxRate,'kd');
    
    %Find the maximum on the curve when the scaling law is not satisfied.
    rates_CO_scaling1 = reshape(sum(sum(mean(rates_MRC_CO_distributed_analytical(:,:,:,:,4),3),2),1),[length(Nantennas) 1])/(Ksite);
    [maxRate,ind] = max(rates_CO_scaling1);
    plot(Nantennas(ind),maxRate,'bd');
    
    legend('Ideal Hardware','Non-Ideal Hardware: SLOs','Non-Ideal Hardware: CLO','Location','SouthEast');
    xlabel('Number of Receive Antennas per Cell');
    ylabel('Average Rate per UE [bit/channel use]');
    
    axis([0 max(Nantennas) 0 5.5]);
    
elseif simulateFigure == 8
    
    if runMonteCarlo_MMSE == true
        
        figure(8); hold on; box on;
        
        plot(Nantennas,reshape(sum(sum(mean(rates_MMSE_SOs_distributed_numerical(:,:,:,:,1),3),2),1),[length(Nantennas) 1])/(Ksite),'r','LineWidth',1);
        
        for ex = 2:length(scalingExponents)
            plot(Nantennas,reshape(sum(sum(mean(rates_MMSE_SOs_distributed_numerical(:,:,:,:,ex),3),2),1),[length(Nantennas) 1])/(Ksite),'k--','LineWidth',1);
            plot(Nantennas,reshape(sum(sum(mean(rates_MMSE_CO_distributed_numerical(:,:,:,:,ex),3),2),1),[length(Nantennas) 1])/(Ksite),'b-.','LineWidth',1);
        end
        
        legend('Ideal Hardware','Non-Ideal Hardware: SLOs','Non-Ideal Hardware: CLO','Location','SouthEast')
        xlabel('Number of Receive Antennas per Cell')
        ylabel('Average Rate per UE [bit/channel use]')
        
    end
    
end
