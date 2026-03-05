%==========================================================================
% Offline MPC Ingredients (LMI) for 8-State Vehicle Model
%
% x = [px, py, psi, v, r, beta, a, delta]      (nx=8)
% u = [delta_dot, a_cmd]                       (nu=2)
%
% Goal:
%   1) Terminal Lyapunov/LMI design on reduced subsystem xr
%   2) Tube tightening design (Pdelta, Kdelta) on reduced subsystem
%   3) Export all parameters in Python-compatible flattened layout
%   4) Provide sanity plots (reduced linear + nonlinear saturated sim)
%
% Notes on Kdelta:
%   Kdelta is typically used as ancillary feedback on the deviation/error:
%       u = u_nom + Kdelta*(x - x_nom)   or u = Kdelta*e
%   It is NOT meant as "nominal controller u=Kdelta*x".
%   Therefore, reduced tests below simulate Kdelta mainly as error feedback.
%
% Python load conventions (examples/vehicle_8state_obs/samplempc_obs.py):
%   Q,P,Pdelta,R:  NumPy reshape(vec,(n,n)) without transpose  -> write row-major
%   K,Kinit,Kdelta: reshape(vec,(nx,nu)).T => store MATLAB as (nu x nx) flat
%   Lx: reshape(vec,(nx,nconstr)).T => store MATLAB as (nconstr x nx)
%   Lu: reshape(vec,(nu,nconstr)).T => store MATLAB as (nconstr x nu)
%   Ls: reshape(vec,(1,nconstr)).T => store MATLAB as (nconstr x 1)
%
% Robust design on reduced xr (default): [psi v r beta a delta] => idx_r=[3..8]
% Positions px,py excluded from reduced LMI, but included in full P/Pdelta.
%==========================================================================

clear; clc; close all;

%% ========================================================================
% USER OPTIONS
%% ========================================================================
ASK_BEFORE_SAVE = true;
DEFAULT_OUTDIR  = "mpc_parameter_8state";

PLOT_FIGURES    = true;
RUN_FULL_NL_SIM = true;

% Start debugging with disturbance OFF; enable later once nominal works
USE_INPUT_DIST_NL = false;

% Reduced linear test: use saturation to avoid misleading huge inputs
USE_SAT_IN_REDUCED_TEST = true;

% If tightening LMI fails: fallback to something usable
ALLOW_TIGHTENING_FALLBACK = true;

rng(1);

%% ========================================================================
% DIMENSIONS
%% ========================================================================
nx = 8;
nu = 2;

% Reduced subsystem xr: [psi v r beta a delta]
idx_r = [3 4 5 6 7 8];
nxr = numel(idx_r);

S_r   = zeros(nxr,nx); S_r(:,idx_r) = eye(nxr);  % xr = S_r*x
S_pos = zeros(2,nx);   S_pos(:,1:2) = eye(2);    % pos = S_pos*x

%% ========================================================================
% MODEL PARAMETERS (MUST MATCH your f.py)
%% ========================================================================
Lwb = 2.5657;
lf  = 1.35;
lr  = Lwb - lf;

g   = 9.81;
m   = 1679 + 82 + 95;       % example
Iz  = 2458;
h   = 0.48;
mu  = 1.0;

C_Sf = 21.92/1.0489;
C_Sr = 21.92/1.0489;

eps_v = 0.5;               % v_safe = sqrt(v^2 + eps_v^2)
tau_a = 0.25;              % a_dot = (a_cmd - a)/tau_a

%% ========================================================================
% BOUNDS (inputs) + state bound on delta
%% ========================================================================
delta_dot_max = 0.60;         % rad/s
a_max         = 3.2;          % m/s^2
a_min         = -6.0;         % m/s^2
delta_max     = 25*pi/180;    % rad (state bound)

% (optional) plausible speed bounds for NL sanity sim
v_min = 0.5;
v_max = 25.0;

%% ========================================================================
% ROBUST VERTICES (over speed)
%% ========================================================================
v_vertices = [0.5, 10.0, 20.0];
Nv = numel(v_vertices);

%% ========================================================================
% TUNING
%% ========================================================================
rho_c = 0.02;                  % contraction rate used in LMIs

Q = diag([2, 2, 0.5, 0.8, 0.2, 0.2, 0.1, 0.2]);
R = diag([10.0, 6.0]);

Qr = Q(idx_r, idx_r);
Rr = R;

Tf = 1.0;
Ppos_term = 1e-3*eye(2);

% Disturbance set for input disturbance (used in tightening LMI)
% Keep small initially; increase later after feasibility is stable.
wbar     = 0.15;
d_ddelta = 0.12*pi/180;
d_acmd   = 0.05;
uw = [ d_ddelta,  d_acmd;
       d_ddelta, -d_acmd;
      -d_ddelta,  d_acmd;
      -d_ddelta, -d_acmd ];

% Position tube shaping (used only to build Pdelta pos block)
gamma_pos_target = 0.25;
Ppos_delta = (1/gamma_pos_target^2)*eye(2);

%% ========================================================================
% PRINT PRE-SOLVE
%% ========================================================================
fprintf('\n================= OFFLINE MPC PARAMS (PRE-SOLVE) =================\n');
fprintf('8-state model: x=[px py psi v r beta a delta], u=[delta_dot a_cmd]\n');
fprintf('Lwb=%.4f, lf=%.4f, lr=%.4f\n', Lwb, lf, lr);
fprintf('Input bounds: delta_dot_max=%.3f rad/s, a in [%.3f, %.3f]\n', delta_dot_max, a_min, a_max);
fprintf('State bound: delta_max=%.1f deg\n', delta_max*180/pi);
fprintf('Vertices v: '); fprintf('%.2f ', v_vertices); fprintf('[m/s]\n');
fprintf('rho_c=%.4f, wbar=%.4f\n', rho_c, wbar);
fprintf('dist corners: d_ddelta=%.3f deg/s, d_acmd=%.3f\n', d_ddelta*180/pi, d_acmd);
fprintf('Q diag: '); fprintf('%.3g ', diag(Q)); fprintf('\n');
fprintf('R diag: '); fprintf('%.3g ', diag(R)); fprintf('\n');
fprintf('Reduced xr indices: '); fprintf('%d ', idx_r); fprintf('\n');
fprintf('==================================================================\n\n');

%% ========================================================================
% SAVE PROMPT
%% ========================================================================
writeout = true;
outdir   = DEFAULT_OUTDIR;

if ASK_BEFORE_SAVE
    resp = input("Save parameters to '" + outdir + "' as .txt? [y/N]: ", "s");
    if strcmpi(strtrim(resp),"y") || strcmpi(strtrim(resp),"yes")
        writeout = true;
        if ~exist(outdir,"dir"), mkdir(outdir); end
        fprintf("-> Will write outputs to %s/\n\n", outdir);
    else
        writeout = false;
        fprintf("-> Will NOT write outputs (will print values instead).\n\n");
    end
else
    if ~exist(outdir,"dir"), mkdir(outdir); end
end

%% ========================================================================
% YALMIP / MOSEK
%% ========================================================================
ops = sdpsettings('solver','mosek','verbose',1,'debug',1);

%% ========================================================================
% Linearization helper: reduced xdot_r = Ar*xr + Br*u
%% ========================================================================
lin_red = @(v0) linearize_reduced_fd( ...
    v0, S_r, nx, nu, ...
    Lwb,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a);

%% ===== DEBUG linearizations =====
for i=1:Nv
    v0 = v_vertices(i);
    [Ar, Br] = lin_red(v0);
    fprintf('\n--- v=%.2f ---\n', v0);
    fprintf('||Ar||_F=%.3e, ||Br||_F=%.3e\n', norm(Ar,'fro'), norm(Br,'fro'));
    fprintf('eig(Ar):\n'); disp(eig(Ar).');
    fprintf('rank(ctrb)= %d / %d\n', rank(ctrb(Ar,Br)), nxr);
    disp('Br rows: [psi v r beta a delta]'); disp(Br);
end

%% ========================================================================
% (1) TERMINAL LMI: find Pr, Kr (common across vertices)
%     Using standard bounded-real type LMI with Q,R shaping
%     Variables:
%       Xr = Pr^{-1}  (SPD)
%       Yr = Kr*Xr
%% ========================================================================
con = [];

Xr = sdpvar(nxr,nxr,'symmetric');
Yr = sdpvar(nu,nxr,'full');

% Bounds prevent pathological scaling (huge Pr -> huge K)
e_reg = 1e-3;
x_min = 1e-6;
x_max = 1e+2;

SQ = sqrtm(Qr + e_reg*eye(nxr));
SR = sqrtm(Rr);

con = [con; Xr >= x_min*eye(nxr)];
con = [con; Xr <= x_max*eye(nxr)];

for i=1:Nv
    [Ar, Br] = lin_red(v_vertices(i));
    AXBY = Ar*Xr + Br*Yr;

    % LMI block (continuous-time decay rho_c)
    ineq = [ AXBY + AXBY' + 2*rho_c*Xr,   (SQ*Xr)',         (SR*Yr)'; ...
             SQ*Xr,                      -eye(nxr),         zeros(nxr,nu); ...
             SR*Yr,                      zeros(nu,nxr),    -eye(nu) ];
    con = [con; ineq <= 0];
end

disp("Solving TERMINAL LMIs for Pr/Kr (reduced xr) ...");
sol1 = optimize(con, -logdet(Xr), ops);
fprintf('\n[Terminal LMI] sol.problem=%d (%s)\n', sol1.problem, yalmiperror(sol1.problem));
if sol1.problem ~= 0
    error("Terminal LMI infeasible/failed.");
end

Xr_num = value(Xr);
Yr_num = value(Yr);

Pr = inv(Xr_num);
Kr = Yr_num * Pr;

fprintf('Kr (nu x nxr) (u = Kr*xr):\n'); disp(Kr);

% Full P and full K (only acts on reduced components)
P = blkdiag(Ppos_term, Pr);     % 2 + nxr = 8
K = zeros(nu,nx);  K(:,idx_r) = Kr;

%% ========================================================================
% (Warm start) LQR gain around v=10 m/s, for initialization in MPC
%% ========================================================================
[Ar0, Br0] = lin_red(10.0);
Kr_lqr = lqr(Ar0, Br0, Qr, 2.0*Rr);      % MATLAB: u = -K*x
Kr_init = -Kr_lqr;                      % we use u = K*x
Kinit = zeros(nu,nx);  Kinit(:,idx_r) = Kr_init;

fprintf('Kr_lqr (MATLAB convention u=-Kx):\n'); disp(Kr_lqr);
fprintf('Kinit (u=Kx):\n'); disp(Kinit);

%% ========================================================================
% (2) Input constraints in "one-sided" normalized form:
%       Lu*u <= 1  (Lx=0 here)
%% ========================================================================
nconstr = 2*nu;  % 4
Lx = zeros(nconstr,nx);
Lu = zeros(nconstr,nu);

Lu(1,1) =  1/delta_dot_max;
Lu(2,2) =  1/a_max;
Lu(3,1) = -1/delta_dot_max;
Lu(4,2) = -1/(-a_min);

% Terminal alpha for feasibility of input constraints inside x'Px <= alpha^2
Lmat = [zeros(nconstr,nx), Lu];
C = zeros(nconstr,1);
Pinv_sqrt = inv(sqrtm(P));
for k=1:nconstr
    C(k) = norm(Pinv_sqrt * [eye(nx), K'] * Lmat(k,:)');
end
alpha = min(1./C);
fprintf('\nTerminal alpha = %.6g\n', alpha);

%% ========================================================================
% (3) TIGHTENING LMI: find Pdelta_r, Kdelta_r
%     Interpretation:
%       - Pdelta defines the tube shape
%       - Kdelta is ancillary feedback for the error dynamics
%
% Key: This LMI can become infeasible if:
%   - wbar too large
%   - coupling too tight
%   - bounds too restrictive
%% ========================================================================
con = [];

Xdr = sdpvar(nxr,nxr,'symmetric');   % Xdr = Pdelta_r^{-1}
Ydr = sdpvar(nu,nxr,'full');         % Ydr = Kdelta_r*Xdr

xd_min = 1e-6;
xd_max = 1e+3;
con = [con; Xdr >= xd_min*eye(nxr)];
con = [con; Xdr <= xd_max*eye(nxr)];

for i=1:Nv
    [Adr, Bdr] = lin_red(v_vertices(i));
    AXBY = Adr*Xdr + Bdr*Ydr;
    con = [con; (AXBY + AXBY' + 2*rho_c*Xdr) <= 0];
end

% Relaxed coupling (do NOT force Pdelta_r >= Pr too tightly)
kappa = 1e3;  % try 1e2..1e5
con = [con; Xdr <= kappa * Xr_num];

% Conservative input-compatibility constraint
for j=1:nconstr
    ineq = [1,               Lu(j,:)*Ydr; ...
            (Lu(j,:)*Ydr)',  Xdr         ];
    con = [con; ineq >= 0];
end

% Disturbance energy constraints (approx, using Br at max v)
[~, Br_worst] = lin_red(max(v_vertices));
dwr = (Br_worst*uw')';  % each row is a direction
for j=1:size(dwr,1)
    ineq = [Xdr,       dwr(j,:)'; ...
            dwr(j,:),  wbar^2    ];
    con = [con; ineq >= 0];
end

disp("Solving TIGHTENING LMIs for Pdelta/Kdelta (reduced xr) ...");
obj = trace(Xdr) + 1e-6*norm(Ydr,'fro');
sol2 = optimize(con, obj, ops);
fprintf('\n[Tightening LMI] sol.problem=%d (%s)\n', sol2.problem, yalmiperror(sol2.problem));

if sol2.problem ~= 0
    if ~ALLOW_TIGHTENING_FALLBACK
        error("Tightening LMI infeasible/failed.");
    end
    warning("Tightening LMI infeasible. Fallback: Pdelta=P, Kdelta=Kinit.");
    Pdelta_r = Pr;
    Kdelta_r = Kr_init;
else
    Xdr_num = value(Xdr);
    Ydr_num = value(Ydr);
    Pdelta_r = inv(Xdr_num);
    Kdelta_r = Ydr_num * Pdelta_r;
end

Pdelta = blkdiag(Ppos_delta, Pdelta_r);
Kdelta = zeros(nu,nx);  Kdelta(:,idx_r) = Kdelta_r;

% Ls coefficients cj (tightening coefficients)
cj = zeros(nconstr,1);
Pdelta_inv_sqrt = inv(sqrtm(Pdelta));
for i=1:nconstr
    cj(i) = norm(Pdelta_inv_sqrt * [eye(nx), Kdelta'] * [zeros(1,nx), Lu(i,:)]');
end

alpha_s = norm(sqrtm(P) / sqrtm(Pdelta));
fprintf('\nalpha_s = %.6g\n', alpha_s);

%% ========================================================================
% (4) Obstacle tightening coefficient gamma_pos
%% ========================================================================
Mpos = S_pos * (Pdelta \ eye(nx)) * S_pos';
Mpos = (Mpos + Mpos')/2;
gamma_pos = sqrt(max(eig(Mpos)));

s_max = (1 - exp(-rho_c * Tf)) / rho_c * wbar;
r_tube_max = gamma_pos * s_max;

fprintf('\nObstacle tightening: gamma_pos = %.6g\n', gamma_pos);
fprintf('s_max(Tf)=%.6g, r_tube_max=%.6g [m]\n', s_max, r_tube_max);

%% ========================================================================
% (5) Export pack
%% ========================================================================
OUT = struct();
OUT.P         = P;
OUT.Q         = Q;
OUT.R         = R;
OUT.K         = K;
OUT.Kinit     = Kinit;
OUT.Kdelta    = Kdelta;
OUT.Pdelta    = Pdelta;
OUT.Lx        = Lx;
OUT.Lu        = Lu;
OUT.Ls        = cj;
OUT.alpha     = alpha;
OUT.alpha_s   = alpha_s;
OUT.wbar      = wbar;
OUT.rho_c     = rho_c;
OUT.Tf        = Tf;
OUT.gamma_pos = gamma_pos;
OUT.r_tube_max = r_tube_max;

if writeout
    if ~exist(outdir,"dir"), mkdir(outdir); end

    % Square matrices: write row-major using transpose then reshape
    writematrix(reshape(round(OUT.Q.',6),1,[]),      fullfile(outdir,'Q.txt'));
    writematrix(reshape(round(OUT.P.',6),1,[]),      fullfile(outdir,'P.txt'));
    writematrix(reshape(round(OUT.R.',6),1,[]),      fullfile(outdir,'R.txt'));
    writematrix(reshape(round(OUT.Pdelta.',6),1,[]), fullfile(outdir,'Pdelta.txt'));

    % Gains stored as (nu x nx)
    writematrix(reshape(round(OUT.K,6),1,[]),        fullfile(outdir,'K.txt'));
    writematrix(reshape(round(OUT.Kinit,6),1,[]),    fullfile(outdir,'Kinit.txt'));
    writematrix(reshape(round(OUT.Kdelta,6),1,[]),   fullfile(outdir,'Kdelta.txt'));

    % Constraints store as (nconstr x *)
    writematrix(reshape(round(OUT.Lx,6),1,[]),       fullfile(outdir,'Lx.txt'));
    writematrix(reshape(round(OUT.Lu,6),1,[]),       fullfile(outdir,'Lu.txt'));
    writematrix(reshape(round(OUT.Ls,6),1,[]),       fullfile(outdir,'Ls.txt'));

    % Scalars
    writematrix(round(OUT.alpha,6),                  fullfile(outdir,'alpha.txt'));
    writematrix(round(OUT.alpha_s,6),                fullfile(outdir,'alpha_s.txt'));
    writematrix(round(OUT.wbar,6),                   fullfile(outdir,'wbar.txt'));
    writematrix(round(OUT.rho_c,6),                  fullfile(outdir,'rho_c.txt'));
    writematrix(round(OUT.Tf,6),                     fullfile(outdir,'Tf.txt'));
    writematrix(round(OUT.gamma_pos,6),              fullfile(outdir,'gamma_pos.txt'));
    writematrix(round(OUT.r_tube_max,6),             fullfile(outdir,'r_tube_max.txt'));

    fprintf("\nExported (Python-compatible) to %s/\n", outdir);
end

%% ========================================================================
% (6) PLOTS: Reduced linear closed-loop sanity check
%   IMPORTANT:
%     - reduced simulation is NOT the full vehicle simulation
%     - if you apply Kdelta as u=Kdelta*x, it can look insane
%     - therefore:
%         * Compare Kinit vs K as nominal controllers
%         * Show Kdelta only as "error feedback" for e dynamics (optional)
%% ========================================================================
if PLOT_FIGURES
    fprintf('\n================ REDUCED LINEAR TESTS ================\n');

    v_test = 10.0;
    Ts = 0.01; Tend = 5.0;

    % xr = [psi v r beta a delta]
    xr0_list = [ 0.30  8.0  0.00  0.00  0.00  0.00;
                -0.25  6.0  0.10  0.05  0.00  0.05;
                 0.15 -2.0  0.00 -0.05  0.00 -0.02 ]';

    % Controllers on reduced coordinates
    Kr_init = Kinit(:,idx_r);
    Kr_lmi  = K(:,idx_r);
    Kr_del  = Kdelta(:,idx_r);

    % Plot style config (easy to distinguish)
    ST = plot_styles();

    % Tiled layout: 2x3 signals
    figure('Name','Reduced linear responses (with sat)','Color','w');
    tl = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
    title(tl, sprintf('Reduced linear @ v=%.1f m/s  (sat=%d)', v_test, USE_SAT_IN_REDUCED_TEST));

    ax1 = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('\psi [rad]'); title('\psi(t)');
    ax2 = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('v [m/s]');   title('v(t)');
    ax3 = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('r [rad/s]'); title('r(t)');
    ax4 = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('\beta [rad]'); title('\beta(t)');
    ax5 = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('a [m/s^2]'); title('a(t)');
    ax6 = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('\delta [rad]'); title('\delta(t)');

    legend_entries = {};

    for ic = 1:size(xr0_list,2)
        xr0 = xr0_list(:,ic);

        % Nominal closed-loop with Kinit
        [t, xr_hist, u_hist] = sim_reduced_cl_sat(Kr_init, v_test, xr0, Ts, Tend, lin_red, ...
            USE_SAT_IN_REDUCED_TEST, delta_dot_max, a_min, a_max);
        plot(ax1,t,xr_hist(1,:),ST.kinit{:});
        plot(ax2,t,xr_hist(2,:),ST.kinit{:});
        plot(ax3,t,xr_hist(3,:),ST.kinit{:});
        plot(ax4,t,xr_hist(4,:),ST.kinit{:});
        plot(ax5,t,xr_hist(5,:),ST.kinit{:});
        plot(ax6,t,xr_hist(6,:),ST.kinit{:});

        % Nominal closed-loop with K (terminal)
        [t, xr_hist, u_hist] = sim_reduced_cl_sat(Kr_lmi, v_test, xr0, Ts, Tend, lin_red, ...
            USE_SAT_IN_REDUCED_TEST, delta_dot_max, a_min, a_max);
        plot(ax1,t,xr_hist(1,:),ST.k{:});
        plot(ax2,t,xr_hist(2,:),ST.k{:});
        plot(ax3,t,xr_hist(3,:),ST.k{:});
        plot(ax4,t,xr_hist(4,:),ST.k{:});
        plot(ax5,t,xr_hist(5,:),ST.k{:});
        plot(ax6,t,xr_hist(6,:),ST.k{:});

        % Kdelta as error feedback (simulate e dynamics: e_dot=(A+BKdelta)e)
        % We just reuse the same sim with initial e0=xr0 to show contraction.
        [t, e_hist, u_hist] = sim_reduced_cl_sat(Kr_del, v_test, xr0, Ts, Tend, lin_red, ...
            USE_SAT_IN_REDUCED_TEST, delta_dot_max, a_min, a_max);
        plot(ax1,t,e_hist(1,:),ST.kdelta{:});
        plot(ax2,t,e_hist(2,:),ST.kdelta{:});
        plot(ax3,t,e_hist(3,:),ST.kdelta{:});
        plot(ax4,t,e_hist(4,:),ST.kdelta{:});
        plot(ax5,t,e_hist(5,:),ST.kdelta{:});
        plot(ax6,t,e_hist(6,:),ST.kdelta{:});

        % only add legend once (avoid 100 entries)
        if ic==1
            legend_entries = {'Kinit','K (terminal)','Kdelta (error fb)'};
        end
    end

    % One legend for the whole figure (put into last axis)
    legend(ax6, legend_entries, 'Location','best');
end

%% ========================================================================
% (7) Nonlinear sanity sim: compare Kinit vs K
%     (Kdelta not used here as nominal controller)
%% ========================================================================
if RUN_FULL_NL_SIM
    fprintf('\n================ NONLINEAR SANITY SIM ================\n');

    x0 = zeros(nx,1);
    x0(3) = 0.25;   % psi
    x0(4) = 8.0;    % v

    Ts_nl = 0.01; Tend_nl = 6.0;

    [t1, x1, u1] = sim_nonlinear_8(Kinit, x0, Ts_nl, Tend_nl, ...
        Lwb,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a, ...
        delta_dot_max, a_min, a_max, delta_max, v_min, v_max, uw, USE_INPUT_DIST_NL);

    [t2, x2, u2] = sim_nonlinear_8(K,     x0, Ts_nl, Tend_nl, ...
        Lwb,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a, ...
        delta_dot_max, a_min, a_max, delta_max, v_min, v_max, uw, USE_INPUT_DIST_NL);

    if PLOT_FIGURES
        ST = plot_styles();

        figure('Name','Nonlinear compare','Color','w');
        tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
        title(tl,'Nonlinear saturated sim (RK4 + bounds + anti-windup)');

        ax = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('\psi [rad]'); title('\psi(t)');
        plot(t1, x1(3,:), ST.kinit{:}); plot(t2, x2(3,:), ST.k{:}); legend('Kinit','K');

        ax = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('v [m/s]'); title('v(t)');
        plot(t1, x1(4,:), ST.kinit{:}); plot(t2, x2(4,:), ST.k{:}); legend('Kinit','K');

        ax = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('\delta\_dot [rad/s]'); title('\delta\_dot(t)');
        stairs(t1(1:end-1), u1(1,:), ST.kinit{:});
        stairs(t2(1:end-1), u2(1,:), ST.k{:});
        yline(delta_dot_max,'--'); yline(-delta_dot_max,'--');
        legend('Kinit','K');

        ax = nexttile; grid on; hold on; xlabel('t [s]'); ylabel('a\_cmd [m/s^2]'); title('a\_cmd(t)');
        stairs(t1(1:end-1), u1(2,:), ST.kinit{:});
        stairs(t2(1:end-1), u2(2,:), ST.k{:});
        yline(a_max,'--'); yline(a_min,'--');
        legend('Kinit','K');
    end
end

fprintf('\n================ DONE ================\n');

%% ========================================================================
% LOCAL FUNCTIONS
%% ========================================================================

function ST = plot_styles()
% Distinct styles without manually specifying colors (MATLAB default color order)
% We vary line style + width; colors auto-cycle by plot order.
    ST.kinit  = {'LineWidth',2.0,'LineStyle','-'};
    ST.k      = {'LineWidth',2.0,'LineStyle','--'};
    ST.kdelta = {'LineWidth',2.0,'LineStyle',':'};
end

function [Ar, Br] = linearize_reduced_fd(v0, S_r, nx, nu, ...
    L,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a)

    % Operating point (straight driving, zero angles)
    xstar = zeros(nx,1);
    xstar(4) = v0;          % v
    ustar = zeros(nu,1);    % [delta_dot; a_cmd]

    ffull = @(x,u) vehicle8_f(x,u,L,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a);

    hfd = 1e-6;
    fx0 = ffull(xstar, ustar);

    A_full = zeros(nx,nx);
    B_full = zeros(nx,nu);

    for i=1:nx
        dx = zeros(nx,1); dx(i)=hfd;
        A_full(:,i) = (ffull(xstar+dx, ustar) - fx0)/hfd;
    end
    for j=1:nu
        du = zeros(nu,1); du(j)=hfd;
        B_full(:,j) = (ffull(xstar, ustar+du) - fx0)/hfd;
    end

    % Reduced projection: assume variation only in reduced coordinates
    SrT = S_r';
    Ar = S_r * A_full * SrT;
    Br = S_r * B_full;
end

function xdot = vehicle8_f(x,u,L,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a)
    % x = [px py psi v r beta a delta]
    % u = [delta_dot a_cmd]
    px=x(1); py=x(2); psi=x(3); v=x(4); r=x(5); beta=x(6); a=x(7); delta=x(8); %#ok<NASGU>
    delta_dot=u(1); a_cmd=u(2);

    v_safe = sqrt(v*v + eps_v*eps_v);

    px_dot  = v*cos(beta+psi);
    py_dot  = v*sin(beta+psi);
    psi_dot = r;
    v_dot   = a;
    a_dot   = (a_cmd - a)/tau_a;
    delta_dot_state = delta_dot;

    r_dot = (-mu*m/(v_safe*Iz*L) * (lf^2*C_Sf*(g*lr-a*h) + lr^2*C_Sr*(g*lf+a*h))*r ...
           +  mu*m/(Iz*L) * (lr*C_Sr*(g*lf+a*h) - lf*C_Sf*(g*lr-a*h))*beta ...
           +  mu*m/(Iz*L) * (lf*C_Sf*(g*lr-a*h))*delta);

    beta_dot = ( (mu/(v_safe*v_safe*L)*(C_Sr*(g*lf+a*h)*lr - C_Sf*(g*lr-a*h)*lf) - 1.0)*r ...
               - mu/(v_safe*L)*(C_Sr*(g*lf+a*h) + C_Sf*(g*lr-a*h))*beta ...
               + mu/(v_safe*L)*(C_Sf*(g*lr-a*h))*delta );

    xdot = [px_dot; py_dot; psi_dot; v_dot; r_dot; beta_dot; a_dot; delta_dot_state];
end

function [t, xr_hist, u_hist] = sim_reduced_cl_sat(Kr, v0, xr0, Ts, Tend, lin_red, ...
    use_sat, delta_dot_max, a_min, a_max)

    % Simulate reduced linear model: xdot = Ar*x + Br*u, u=Kr*x
    [Ar, Br] = lin_red(v0);

    N = round(Tend/Ts);
    t = (0:N)*Ts;

    nxr = numel(xr0);
    xr_hist = zeros(nxr, N+1);
    u_hist  = zeros(size(Kr,1), N);

    xr_hist(:,1) = xr0;

    for k=1:N
        u = Kr * xr_hist(:,k);

        if use_sat
            u(1) = min(max(u(1), -delta_dot_max), delta_dot_max);
            u(2) = min(max(u(2),  a_min),        a_max);
        end

        u_hist(:,k) = u;
        xdot = Ar*xr_hist(:,k) + Br*u;
        xr_hist(:,k+1) = xr_hist(:,k) + Ts*xdot;
    end
end

function [t, x, u_hist] = sim_nonlinear_8(Kfull, x0, Ts, Tend, ...
    L,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a, ...
    delta_dot_max, a_min, a_max, delta_max, v_min, v_max, uw, use_dist)

    N = round(Tend/Ts);
    t = (0:N)*Ts;

    nx = numel(x0);
    x = zeros(nx, N+1);
    u_hist = zeros(2, N);
    x(:,1) = x0;

    f = @(xx,uu) vehicle8_f(xx, uu, L,lf,lr,g,m,Iz,h,mu,C_Sf,C_Sr,eps_v,tau_a);

    for k=1:N
        u = Kfull * x(:,k);

        % Optional bounded input disturbance (debug later)
        if use_dist
            idx = randi(size(uw,1));
            u = u + uw(idx,:)';
        end

        % Anti-windup / guard: if delta is saturated, do not push further
        if x(8,k) >= delta_max && u(1) > 0
            u(1) = 0;
        elseif x(8,k) <= -delta_max && u(1) < 0
            u(1) = 0;
        end

        % Input saturation
        u(1) = min(max(u(1), -delta_dot_max), delta_dot_max);
        u(2) = min(max(u(2),  a_min),        a_max);
        u_hist(:,k) = u;

        % RK4 integration
        k1 = f(x(:,k), u);
        k2 = f(x(:,k) + 0.5*Ts*k1, u);
        k3 = f(x(:,k) + 0.5*Ts*k2, u);
        k4 = f(x(:,k) + Ts*k3, u);
        x(:,k+1) = x(:,k) + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);

        % State bounds
        x(8,k+1) = min(max(x(8,k+1), -delta_max), +delta_max); % delta bound
        x(4,k+1) = min(max(x(4,k+1), v_min), v_max);           % v bound
        x(3,k+1) = wrapToPi(x(3,k+1));                         % psi wrap
    end
end