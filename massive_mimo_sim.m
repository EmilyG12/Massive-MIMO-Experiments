%% ========================================================================
%  MASSIVE MIMO MMSE DETECTION: EXACT vs APPROXIMATE MATRIX INVERSION
%  ========================================================================
%  PURPOSE
%    Linear MMSE detection needs the solution of  G*x = b  where G is a
%    U x U Hermitian positive-definite matrix. This script compares ways of
%    solving that system:
%
%      EXACT       : direct inversion, Cholesky decomposition, QR decomposition
%      APPROXIMATE : Neumann series (few terms), conjugate gradient (few iterations)
%
%  FIGURES PRODUCED (4 in total)
%    Figure 1 : BER vs SNR                    (fixed N and U)
%    Figure 2 : BER vs N/U ratio              (fixed SNR)
%    Figure 3 : conditioning of G vs N/U      (fixed SNR)
%    Figure 4 : runtime vs number of users U  (one graph, whole detection step)
%
%  NOTATION
%    N = number of base-station (BS) antennas   (rows of H)
%    U = number of single-antenna users          (columns of H)
%    T = number of symbol vectors sent per coherence block
%    H = N x U channel matrix,  x = U x 1 transmitted symbols,
%    y = N x 1 received signal, n = N x 1 noise
%
%  NOTE: local functions (the methods) must stay at the END of this file.
%  ========================================================================
clear; clc; close all;
rng(1);                              % fixed random seed -> reproducible results

%% ========================================================================
%  SECTION 1: SIMULATION SETUP
%  ========================================================================
%  --- SYSTEM MODEL -------------------------------------------------------
%    y = H*x + n                                   (uplink, one symbol vector)
%
%  CHANNEL MATRIX H  (i.i.d. Rayleigh fading)
%    Every entry H(i,j) is an independent circularly-symmetric complex
%    Gaussian CN(0,1) random variable:  H(i,j) = (randn + 1j*randn)/sqrt(2).
%    Its magnitude is Rayleigh distributed (no line of sight, rich scattering)
%    and every antenna-user link is statistically independent of the others.
%    BLOCK FADING: H is drawn once per "coherence block" and stays constant for
%    the T symbol vectors sent in that block, then a new independent H is drawn.
%    This is why the expensive step (forming/factoring G) is done once per
%    channel while the cheap step (solving) is repeated for each symbol vector.
%
%  MODULATION (QPSK, Gray mapped, unit symbol energy Es = 1)
%    Each symbol carries 2 bits (bI, bQ):  x = ((1-2*bI) + 1j*(1-2*bQ))/sqrt(2)
%    i.e. bit 0 -> +1/sqrt(2), bit 1 -> -1/sqrt(2) on each of the I and Q axes.
%    The detector makes a hard decision on the sign of the real and imaginary
%    parts of the estimate: negative -> bit 1, positive -> bit 0.
%
%  NOISE
%    n(i) ~ CN(0, sigma^2), i.e. real and imaginary parts each N(0, sigma^2/2).
%
%  MMSE DETECTOR (what every method is trying to compute)
%    xhat = (H'H + sigma^2 I)^(-1) H' y  =  G^(-1) * b,
%    with G = H'H + sigma^2*I   (U x U, Hermitian positive definite)
%         b = H'*y              (U x 1, "matched-filter output")
%    ' is the conjugate (Hermitian) transpose in MATLAB.
%
%  SNR DEFINITION ("SNR after array gain")
%    snrEff = 10*log10( N*Es/sigma^2 )   ->   sigma^2 = N * 10^(-snrEff/10)
%    N*Es/sigma^2 is the matched-filter SNR per user (the BS array adds a gain
%    of N). Defining SNR this way means that changing N in Experiment 2 does not
%    change the SNR, so differences in BER come from interference and
%    conditioning rather than from extra antenna gain. State this definition
%    in the report. (Es/sigma^2 per antenna = snrEff - 10*log10(N) dB.)
%
%  NUMBER OF TRIALS (Monte Carlo)
%    For every plotted point:  nBlk independent channel realisations
%                              x T symbol vectors per realisation
%                              x U users x 2 bits per QPSK symbol
%    e.g. 1000 x 14 x 16 x 2 = 448,000 bits. The smallest measurable BER is
%    about 1/448000 (about 2e-6); BERs below ~1e-4 are noisy (fewer than ~45
%    errors). Increase nBlk for smoother curves at low BER.
%
%  FAIR COMPARISON
%    For every block, ALL methods receive exactly the same H, x and noise. The
%    differences between curves are therefore caused only by the method.
%  ------------------------------------------------------------------------
PREC     = 'double';        % 'double' or 'single' (single mimics limited-precision hardware)
T        = 14;              % symbol vectors per coherence block
nBlk     = 1000;            % channel realisations (blocks) per plotted point

% Experiment 1 (BER vs SNR): fixed system size
N1       = 64;              % BS antennas        (N/U = 4: approximate methods visibly differ)
U1       = 16;              % users
snrEff   = 0:2:12;          % SNR axis in dB (after array gain). Adjust so the exact-method
                            % BER spans roughly 1e-1 down to 1e-4.

% Experiment 2 (BER and conditioning vs N/U): fixed SNR, vary N
U2       = 16;              % users
ratios   = [1.25 1.5 2 3 4 6 8 12 16];   % N/U values  ->  N = 20 ... 256 antennas
snrFix   = 10;              % fixed SNR in dB (after array gain)

% Approximate-method settings (these are the "accuracy vs cost" knobs)
neuTerms = [2 3];           % Neumann series: number of terms used
cgIters  = [2 3];           % conjugate gradient: number of iterations used

% Experiment 3 (runtime)
Ulist    = 16:16:256;       % number of users
Nt       = 512;             % BS antennas (kept fixed and >= max(Ulist))
includeMatrixFree = true;   % also time a CG that never forms G (see solve_cgmf)

%% ========================================================================
%  SECTION 2: LIST OF METHODS
%  ========================================================================
%  Every method is split into two functions, matching how a real receiver works:
%    prep_X(H, s2)    : per-CHANNEL work (done once per coherence block):
%                       forms G and factorises it / prepares the iteration.
%    solve_X(F, H, Y) : per-SYMBOL work (done for each received vector y):
%                       forms b = H'*y and solves G*x = b using the stored factors F.
%  Each method's maths is described next to its function at the end of the file.
names = {'Direct inversion', 'Cholesky', 'QR'};
prep  = {@prep_inv, @prep_chol, @prep_qr};
solve = {@solve_inv, @solve_chol, @solve_qr};
for kk = neuTerms
    names{end+1} = sprintf('Neumann (%d terms)', kk);
    prep{end+1}  = @prep_gram;
    solve{end+1} = @(F,H,Y) solve_neu(F,H,Y,kk);
end
for kk = cgIters
    names{end+1} = sprintf('CG (%d iterations)', kk);
    prep{end+1}  = @prep_gram;
    solve{end+1} = @(F,H,Y) solve_cg(F,H,Y,kk);
end
nM  = numel(names);
% Plot styling: exact methods solid/dark, approximate methods warm colours
mk  = {'o-','s--','^:','d-','v-','p--','x:','*-'};
col = [0.00 0.45 0.74;  0.85 0.33 0.10;  0.47 0.67 0.19;
       0.93 0.69 0.13;  0.49 0.18 0.56;  0.30 0.75 0.93;  0.64 0.08 0.18;  0.25 0.25 0.25];

%% ========================================================================
%  SECTION 3: EXPERIMENT 1 - BER vs SNR
%  ========================================================================
%  HOW IT IS PERFORMED
%    For each SNR point:
%      1. Compute sigma^2 = N1*10^(-snr/10).
%      2. Repeat nBlk times (see runBER below):
%           a. Draw a new i.i.d. Rayleigh H (N1 x U1).
%           b. Draw random bits, map to QPSK symbols X (U1 x T).
%           c. Form Y = H*X + noise (N1 x T).
%           d. For each method: run prep (once), then solve for all T columns
%              of Y, make hard decisions, count bit errors against the sent bits.
%      3. BER = total bit errors / total bits sent.
%  EXPECTED RESULT
%    The exact methods (inversion, Cholesky, QR) solve the same equation exactly,
%    so their curves coincide. Neumann/CG are truncated, so their BER is slightly
%    higher, and improves as terms/iterations increase.
fprintf('Experiment 1: BER vs SNR (N=%d, U=%d, %s precision)\n', N1, U1, PREC);
ber1 = zeros(nM, numel(snrEff));
for si = 1:numel(snrEff)
    s2 = N1*10^(-snrEff(si)/10);
    [ber1(:,si), nFail, ~] = runBER(N1, U1, s2, nBlk, T, prep, solve, PREC);
    fprintf('  snrEff = %2d dB:  ', snrEff(si));
    pairs = [names; num2cell(ber1(:,si).')];       % 2 x nM cell: method name, BER value
    fprintf('%s = %.2e   ', pairs{:});             % {:} expands the cell into arguments
    fprintf('\n');
    if any(nFail), fprintf('    NaN/Inf outputs per method: %s\n', mat2str(nFail')); end
end
plotBER(snrEff, ber1, names, mk, col, 'SNR after array gain (dB)', ...
        sprintf('BER vs SNR, N=%d, U=%d, QPSK, i.i.d. Rayleigh', N1, U1), 'Fig1 BER vs SNR');

%% ========================================================================
%  SECTION 4: EXPERIMENT 2 - BER AND CONDITIONING vs N/U
%  ========================================================================
%  HOW IT IS PERFORMED
%    U2 is kept fixed and the number of antennas N = round(ratio*U2) is varied.
%    The SNR (after array gain) is fixed at snrFix for every ratio. For each N
%    the same Monte Carlo procedure as Experiment 1 is run, and in addition the
%    condition number of G = H'H + sigma^2*I is averaged over the blocks.
%  WHY N/U MATTERS
%    G = H'H + sigma^2*I. For large N/U, H'H is approximately N*I (the columns of
%    H are nearly orthogonal, "channel hardening"), so G is almost diagonal and
%    well conditioned (cond(G) -> 1). For N/U close to 1, G has widely spread
%    eigenvalues, so cond(G) is large. Neumann and CG converge fast when G is
%    close to diagonal / well conditioned, and slowly otherwise.
%  EXPECTED RESULT
%    Exact methods coincide. Neumann/CG BER is higher at small N/U and approaches
%    the exact methods as N/U grows. cond(G) decreases as N/U increases.
Nlist = round(ratios*U2);
ber2  = zeros(nM, numel(Nlist));
cond2 = zeros(1, numel(Nlist));
fprintf('\nExperiment 2: BER and cond(G) vs N/U (U=%d, snrEff = %d dB)\n', U2, snrFix);
for i = 1:numel(Nlist)
    s2 = Nlist(i)*10^(-snrFix/10);
    [ber2(:,i), nFail, cond2(i)] = runBER(Nlist(i), U2, s2, nBlk, T, prep, solve, PREC);
    fprintf('  N = %3d (N/U = %5.2f): mean cond(G) = %7.2f\n', Nlist(i), Nlist(i)/U2, cond2(i));
    if any(nFail), fprintf('    NaN/Inf outputs per method: %s\n', mat2str(nFail')); end
end
plotBER(Nlist/U2, ber2, names, mk, col, 'N/U (antennas per user)', ...
        sprintf('BER vs N/U, U=%d, SNR = %d dB (after array gain)', U2, snrFix), 'Fig2 BER vs N over U');

figure('Name', 'Fig3 cond vs N over U');
semilogy(Nlist/U2, cond2, 'ko-', 'LineWidth', 1.5, 'MarkerSize', 7); grid on;
xlabel('N/U (antennas per user)'); ylabel('mean cond(G)');
title(sprintf('Conditioning of G = H^HH + \\sigma^2I vs N/U (U=%d)', U2));

%% ========================================================================
%  SECTION 5: EXPERIMENT 3 - RUNTIME vs NUMBER OF USERS
%  ========================================================================
%  HOW IT IS PERFORMED
%    For each U in Ulist (N = Nt fixed): draw one H (Nt x U) and one block of T
%    received vectors Y (Nt x T). For each method time the WHOLE detection of
%    one coherence block:
%         prep (form G, factorise / set up)  +  solve for all T symbol vectors.
%    This is a single number per method and U, so the results fit on one graph.
%    timeit() runs the function repeatedly and returns the median time, which
%    removes most one-off noise. MATLAB is limited to one thread (if allowed) so
%    that the timing reflects the algorithm and not multi-core scheduling.
%  EXPECTED RESULT
%    Forming G costs about N*U^2 for every method that forms it, so the saving of
%    the approximate methods (they avoid the ~U^3 factorisation) is largest when U
%    is comparable to N. The matrix-free CG never forms G and can be cheaper still.
%    MATLAB loop overhead and optimised library routines (chol, \) also affect
%    the absolute times, so discuss trends rather than exact flop ratios.
names_t = names;  prep_t = prep;  solve_t = solve;
if includeMatrixFree
    names_t{end+1} = sprintf('CG matrix-free (%d it.)', cgIters(1));
    prep_t{end+1}  = @prep_none;
    solve_t{end+1} = @(F,H,Y) solve_cgmf(F,H,Y,cgIters(1));
end
nT   = numel(names_t);
s2t  = 0.1;                                    % any fixed noise variance (timing only)
tDet = zeros(nT, numel(Ulist));
try, nThr = maxNumCompThreads(1); catch, nThr = []; end

fprintf('\nExperiment 3: runtime for one coherence block (N=%d, T=%d)\n', Nt, T);
for ui = 1:numel(Ulist)
    Ut = Ulist(ui);
    Ht = (randn(Nt,Ut) + 1j*randn(Nt,Ut))/sqrt(2);
    Yt = randn(Nt,T)   + 1j*randn(Nt,T);
    for k = 1:nT
        detect = @() solve_t{k}(prep_t{k}(Ht, s2t), Ht, Yt);   % prep once + solve T vectors
        tDet(k,ui) = timeit(detect);
    end
end
if ~isempty(nThr), maxNumCompThreads(nThr); end   % restore thread count

figure('Name', 'Fig4 runtime');
for k = 1:nT
    loglog(Ulist, tDet(k,:), mk{mod(k-1,numel(mk))+1}, 'Color', col(mod(k-1,size(col,1))+1,:), ...
           'LineWidth', 1.5, 'MarkerSize', 6); hold on;
end
grid on; xlabel('Number of users U'); ylabel('Time per coherence block (s)');
title(sprintf('Detection time, N=%d, T=%d symbol vectors per block', Nt, T));
legend(names_t, 'Location', 'northwest');

fprintf('\nFitted log-log slopes (time ~ U^slope):\n');
for k = 1:nT
    p = polyfit(log(Ulist), log(tDet(k,:)), 1);
    fprintf('  %-26s U^%.2f\n', names_t{k}, p(1));
end

% Save figures for LaTeX (MATLAB R2020a or later), for example:
% exportgraphics(figure(1), 'ber_snr.pdf');  exportgraphics(figure(2), 'ber_ratio.pdf');
% exportgraphics(figure(3), 'cond_ratio.pdf'); exportgraphics(figure(4), 'runtime.pdf');

%% ========================================================================
%  LOCAL FUNCTIONS: THE DETECTION METHODS
%  ========================================================================
%  All methods approximate or compute  x = G^(-1) * b,
%  with  G = H'H + s2*I  and  b = H'*Y  (Y has one received vector per column).

% ----- Direct inversion --------------------------------------------------
%  Compute the explicit inverse Ginv = inv(G) once per channel, then x = Ginv*b
%  (a matrix-vector product per symbol vector). Cost is about 2U^3 flops for the
%  inverse; it is also less numerically accurate than solving with a factorisation,
%  which is why it is used here only as the reference "no decomposition" baseline.
function F = prep_inv(H, s2)
    G = H'*H + s2*eye(size(H,2), class(H));
    F.Ginv = inv(G);
end
function X = solve_inv(F, H, Y)
    X = F.Ginv*(H'*Y);
end

% ----- Cholesky decomposition -------------------------------------------
%  G is Hermitian positive definite, so it can be factorised as G = L*L' with L
%  lower triangular (about U^3/3 flops, the cheapest exact factorisation).
%  Solve G*x = b in two triangular solves (each about U^2 flops):
%      L*z  = b    (forward substitution)
%      L'*x = z    (backward substitution)
%  If G is not numerically positive definite (possible in single precision with
%  a badly conditioned channel), chol fails and the output is marked NaN.
function F = prep_chol(H, s2)
    G = H'*H + s2*eye(size(H,2), class(H));
    [L, p] = chol(G, 'lower');
    if p > 0, L = nan(size(G), class(G)); end
    F.L = L;
end
function X = solve_chol(F, H, Y)
    Z = linsolve(F.L,  H'*Y, struct('LT', true));    % forward substitution
    X = linsolve(F.L', Z,    struct('UT', true));    % backward substitution
end

% ----- QR decomposition --------------------------------------------------
%  Factorise G = Q*R with Q unitary (Q'*Q = I) and R upper triangular
%  (about (4/3)U^3 flops with Householder reflections). Then
%      G*x = b   ->   Q*R*x = b   ->   R*x = Q'*b
%  which is solved by backward substitution. Multiplying by Q' preserves norms,
%  so QR is numerically very stable. (Here Q is formed explicitly, which makes
%  the measured time higher than the minimum possible QR cost.)
function F = prep_qr(H, s2)
    G = H'*H + s2*eye(size(H,2), class(H));
    [F.Q, F.R] = qr(G);
end
function X = solve_qr(F, H, Y)
    X = linsolve(F.R, F.Q'*(H'*Y), struct('UT', true));
end

% ----- Shared set-up for the iterative methods (Neumann and CG) ----------
%  Only the Gram matrix is formed (cost about N*U^2); no factorisation is done.
%  The diagonal d = diag(G) is stored for the Neumann series.
function F = prep_gram(H, s2)
    G = H'*H + s2*eye(size(H,2), class(H));
    F.G = G;
    F.d = real(diag(G));
end

% ----- Neumann series (k terms) -------------------------------------------
%  Split G = D + E with D = diag(G) and E = G - D (the off-diagonal part). Then
%      G^(-1) = (I + D^(-1)E)^(-1) D^(-1) = sum_{n=0}^{inf} (-D^(-1)E)^n D^(-1)
%  The series converges when the spectral radius of D^(-1)E is below 1, which
%  holds when G is strongly diagonally dominant (N >> U). Truncating after k terms
%  gives the approximation used here. It is applied to the vector b without
%  ever forming an inverse matrix:
%      t_0 = D^(-1) b,    t_n = -D^(-1) E t_{n-1},    x ~ t_0 + t_1 + ... + t_{k-1}
%  k = 1 is the diagonal approximation; each extra term costs one U x U
%  matrix-vector product (about 2U^2 flops). The line F.G*Tn - F.d.*Tn is E*Tn.
function X = solve_neu(F, H, Y, k)
    Dinv = 1./F.d;
    Tn = Dinv.*(H'*Y);                         % t_0 = D^(-1) b
    X  = Tn;
    for n = 2:k
        Tn = -Dinv.*(F.G*Tn - F.d.*Tn);        % t_n = -D^(-1) E t_{n-1}
        X  = X + Tn;
    end
end

% ----- Conjugate gradient (k iterations) -----------------------------------
%  CG solves G*x = b for Hermitian positive-definite G by minimising the error
%  over a growing set of search directions that are mutually G-conjugate. Starting
%  from x = 0 (so the first residual is r = b and the first direction p = r):
%      alpha = (r'r) / (p'Gp)          step length along p
%      x     = x + alpha*p             update the estimate
%      r     = r - alpha*G*p           update the residual b - G*x
%      beta  = (r_new'r_new)/(r_old'r_old)
%      p     = r + beta*p              new search direction
%  In exact arithmetic CG is exact after U iterations. Stopping after k << U
%  iterations gives an approximation whose error shrinks faster when the
%  eigenvalues of G are clustered (small cond(G)). Each iteration costs one
%  U x U matrix-vector product (about 2U^2 flops). Each column of Y is an
%  independent problem; the scalars alpha and beta are computed per column.
function X = solve_cg(F, H, Y, it)
    B = H'*Y;
    X = zeros(size(B), 'like', B);
    R = B;  P = R;
    rs   = real(sum(conj(R).*R, 1));           % r'r for each column
    tiny = realmin(class(real(B)));            % avoids dividing by zero after convergence
    for n = 1:it
        Gp    = F.G*P;
        alpha = rs ./ max(real(sum(conj(P).*Gp, 1)), tiny);
        X     = X + P.*alpha;
        R     = R - Gp.*alpha;
        rsNew = real(sum(conj(R).*R, 1));
        P     = R + P.*(rsNew ./ max(rs, tiny));
        rs    = rsNew;
    end
end

% ----- Matrix-free conjugate gradient (timing experiment only) ------------
%  Identical algorithm to solve_cg, but G*p is computed as H'*(H*p) + s2*p, so G
%  is never formed (no N*U^2 cost). Each iteration costs about 4*N*U flops
%  instead of 2*U^2, which is cheaper when N is not much larger than U^2/2.
%  It produces the same estimate as solve_cg (up to rounding), so it is not
%  plotted in the BER figures.
function F = prep_none(~, s2)
    F.s2 = s2;
end
function X = solve_cgmf(F, H, Y, it)
    B = H'*Y;
    X = zeros(size(B), 'like', B);
    R = B;  P = R;
    rs   = real(sum(conj(R).*R, 1));
    tiny = realmin(class(real(B)));
    for n = 1:it
        Gp    = H'*(H*P) + F.s2*P;
        alpha = rs ./ max(real(sum(conj(P).*Gp, 1)), tiny);
        X     = X + P.*alpha;
        R     = R - Gp.*alpha;
        rsNew = real(sum(conj(R).*R, 1));
        P     = R + P.*(rsNew ./ max(rs, tiny));
        rs    = rsNew;
    end
end

%% ========================================================================
%  LOCAL FUNCTIONS: MONTE CARLO SIMULATION AND PLOTTING
%  ========================================================================

% runBER: Monte Carlo simulation of the whole link for ONE (N, U, sigma^2) point.
%   Returns the BER of every method, the number of blocks in which a method
%   produced NaN/Inf, and the mean condition number of G = H'H + s2*I.
function [ber, nFail, condG] = runBER(N, U, s2, nBlk, T, prep, solve, PREC)
    nM    = numel(prep);
    err   = zeros(nM, 1);                     % bit-error counters, one per method
    nFail = zeros(nM, 1);
    condG = 0;
    for blk = 1:nBlk
        % (a) i.i.d. Rayleigh channel: every entry CN(0,1), new draw every block
        H  = cast((randn(N,U) + 1j*randn(N,U))/sqrt(2), PREC);

        % (b) random bits and unit-energy QPSK symbols (U users x T symbol vectors)
        bI = randi([0 1], U, T);              % in-phase bits
        bQ = randi([0 1], U, T);              % quadrature bits
        X  = ((1 - 2*bI) + 1j*(1 - 2*bQ))/sqrt(2);

        % (c) received signal: y = H*x + n, with n ~ CN(0, s2)
        Nz = sqrt(s2/2)*(randn(N,T) + 1j*randn(N,T));
        Y  = H*cast(X, PREC) + cast(Nz, PREC);

        % conditioning of the matrix every method has to invert (computed in double)
        condG = condG + cond(double(H'*H) + s2*eye(U));

        % (d) run every method on the SAME H and Y, make hard decisions, count errors
        for k = 1:nM
            F  = prep{k}(H, cast(s2, PREC));      % per-channel work
            Xh = solve{k}(F, H, Y);               % per-symbol work for all T vectors
            if any(~isfinite(Xh(:))), nFail(k) = nFail(k) + 1; end
            % negative real/imag part -> decided bit 1; compare with the sent bits
            err(k) = err(k) + sum(sum((real(Xh) < 0) ~= bI)) ...
                            + sum(sum((imag(Xh) < 0) ~= bQ));
        end
    end
    ber   = err/(2*U*T*nBlk);                 % total bit errors / total bits sent
    condG = condG/nBlk;                       % average over the blocks
end

% plotBER: semilog BER plot with one curve per method.
function plotBER(x, ber, names, mk, col, xl, ttl, figname)
    ber(ber == 0) = NaN;                      % zero BER cannot be drawn on a log axis
    figure('Name', figname);
    for k = 1:numel(names)
        semilogy(x, ber(k,:), mk{mod(k-1,numel(mk))+1}, ...
                 'Color', col(mod(k-1,size(col,1))+1,:), ...
                 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
    end
    grid on; xlabel(xl); ylabel('BER'); title(ttl);
    legend(names, 'Location', 'best');
end