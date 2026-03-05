%==========================================================================
% Offline MPC Ingredients (LMI) for Kinematic Bicycle - ABS psi RANDOM
% + Tube tightening coefficient gamma_pos for OBSTACLE tightening:
%       r_eff(k) = r_safe + gamma_pos * s(k)
%
% Requested changes:
%  (1) Robust Kinit sign/reshape sanity (avoid "a=+10*v" issue)
%  (2) If NOT saved -> print values that would be written to files
%  (3) Tune "looser" to get more feasible NMPC solves
%==========================================================================
clear; clc; close all;

%% ========================================================================
% USER OPTIONS
%% ========================================================================
ASK_BEFORE_SAVE = true;
DEFAULT_OUTDIR  = "mpc_parameter_5";

PLOT_FIGURES    = true;
RUN_FULL_NL_SIM = true;      % nonlinear sanity sim (tan, saturation)
USE_INPUT_DIST  = true;      % bounded input disturbance in nonlinear sim
DO_QR_SWEEP     = false;     % keep off while tuning feasibility

writeout = true;
outdir   = DEFAULT_OUTDIR;

rng(1);  % reproducible tests

%% ========================================================================
% DIMENSIONS + PARAMETERS
%% ========================================================================
nx = 4;      % [px, py, psi, v]
nu = 2;      % [delta, a]
nxr = 2;     % reduced xr=[psi; v]

Lwb = 2.5657;

% bounds
delta_max = 25*pi/180;
a_max     = 3.2;
a_min     = -6.0;

% robust vertices over speed
v_vertices = [0.2, 10.0, 20.0];
Nv = numel(v_vertices);

%% ========================================================================
% LOOSER TUNING (more feasible)
%  - smaller decay rate rho_c (less strict)
%  - larger R (less aggressive K, less saturation)
%  - smaller disturbance energy and corners (smaller tube -> easier feasibility)
%% ========================================================================
rho_c = 0.02;                   % was 0.05 (looser)

Q = diag([1.5, 1.5, 0.25, 0.8]); % was [0.1 0.1 0.5 1.5] (looser)
R = diag([8.0, 4.0]);              % was [4 2] (less aggressive)

e = 0;
Qr = diag([Q(3,3), Q(4,4)]);
Rr = R;

% Tube / tightening knobs (looser)
wbar    = 0.15;          % was 0.5  (smaller s(t) -> smaller tightening)
d_delta = 0.12*pi/180;    % was 0.3 deg
d_a     = 0.05;          % was 0.15
uw = [ d_delta,  d_a;
       d_delta, -d_a;
      -d_delta,  d_a;
      -d_delta, -d_a ];

Tf = 1.0;
% Terminal weight on position (keep tiny, because terminal LMI ignores px/py)
Ppos_term = 1e-3*eye(2);

% NEW: Tube metric position weight (sets gamma_pos!)
gamma_pos_target = 0.25;                 % [m per unit s]  (tune: 0.15..0.5)
Ppos_delta = (1/gamma_pos_target^2)*eye(2);

%% ========================================================================
% PRINT PRE-SOLVE + ASK BEFORE SAVE
%% ========================================================================
fprintf('\n================= OFFLINE MPC PARAMS (PRE-SOLVE) =================\n');
fprintf('Lwb=%.4f\n', Lwb);
fprintf('Bounds: delta_max=%.3f deg, a in [%.3f, %.3f]\n', delta_max*180/pi, a_min, a_max);
fprintf('Vertices v: '); fprintf('%.2f ', v_vertices); fprintf('[m/s]\n');
fprintf('rho_c=%.4f, wbar=%.4f\n', rho_c, wbar);
fprintf('dist corners: d_delta=%.3f deg, d_a=%.3f\n', d_delta*180/pi, d_a);
fprintf('Q diag: '); fprintf('%.3g ', diag(Q)); fprintf('\n');
fprintf('R diag: '); fprintf('%.3g ', diag(R)); fprintf('\n');
fprintf('==================================================================\n\n');

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
    writeout = true;
    if ~exist(outdir,"dir"), mkdir(outdir); end
end

%% ========================================================================
% YALMIP / MOSEK
%% ========================================================================
ops = sdpsettings('solver','mosek','verbose',1,'debug',1);

%% ========================================================================
% (1) TERMINAL LMI on reduced subsystem: common Pr,Kr for vertices
%% ========================================================================
con = [];

Xr = sdpvar(nxr,nxr,'symmetric');    % Xr = Pr^{-1}
Yr = sdpvar(nu,nxr,'full');          % Yr = Kr*Xr

SQ = sqrtm(Qr + e*eye(nxr));
SR = sqrtm(Rr);

for i=1:Nv
    v0 = v_vertices(i);
    Ar = zeros(nxr,nxr);
    Br = [v0/Lwb, 0;
          0,      1];

    AXBY = Ar*Xr + Br*Yr;

    ineq = [ AXBY + AXBY' + 2*rho_c*Xr,   (SQ*Xr)',         (SR*Yr)'; ...
             SQ*Xr,                      -eye(nxr),         zeros(nxr,nu); ...
             SR*Yr,                      zeros(nu,nxr),    -eye(nu) ];
    con = [con; ineq <= 0];
end
con = [con; Xr >= 1e-9*eye(nxr)];

disp("Solving TERMINAL LMIs for Pr/Kr (reduced [psi v]) ...");
sol1 = optimize(con, -trace(Xr), ops);
fprintf('\n[Terminal LMI] sol.problem=%d (%s)\n', sol1.problem, yalmiperror(sol1.problem));
if sol1.problem ~= 0
    error("Terminal LMI infeasible/failed.");
end

Xr_num = value(Xr);
Yr_num = value(Yr);

Pr = inv(Xr_num);
Kr = Yr_num * Pr;      % u = Kr * [psi; v]

fprintf('Kr (2x2) from LMI (u = Kr*[psi;v]):\n'); disp(Kr);

P = blkdiag(Ppos_term, Pr);
K = zeros(nu,nx);  K(:,3:4) = Kr;

%% ========================================================================
% LQR init gain on reduced subsystem (warm start)
%   MATLAB lqr gives u = -K_lqr x
%   we use u = K x  => Kinit = -K_lqr
%% ========================================================================
v0_lqr = 10.0;
Ar = zeros(nxr,nxr);
Br = [v0_lqr/Lwb, 0;
      0,           1];

% Make warm start less aggressive as well:
R_lqr_scale = 2.0;                 % was 1.0
Kr_lqr = lqr(Ar, Br, Qr, Rr*R_lqr_scale);

Kinit = zeros(nu,nx);
Kinit(:,3:4) = -Kr_lqr;

% --- HARD sanity: acceleration feedback must stabilize v around 0
if Kinit(2,4) > 0
    warning("Kinit(2,4)=%.6g is positive (would accelerate for v>0). Flipping sign.", Kinit(2,4));
    Kinit(2,4) = -abs(Kinit(2,4));
end

fprintf('Kr_lqr (MATLAB convention u=-Kx):\n'); disp(Kr_lqr);
fprintf('Kinit (converted to u=Kx):\n'); disp(Kinit);
fprintf('Sanity: Kinit(2,4)=%.6g (should be < 0)\n', Kinit(2,4));

%% ========================================================================
% (2) Constraints format: only input bounds Lx*x + Lu*u <= 1
%% ========================================================================
nconstr = 2*nu;
Lx = zeros(nconstr,nx);
Lu = zeros(nconstr,nu);

Lu(1,1) =  1/delta_max;
Lu(2,2) =  1/a_max;
Lu(3,1) = -1/delta_max;
Lu(4,2) = -1/(-a_min);

% Terminal alpha for u=Kx feasibility inside x'Px <= alpha^2
Lmat = [zeros(nconstr,nx), Lu];
C = zeros(nconstr,1);
Pinv_sqrt = inv(sqrtm(P));
for k=1:nconstr
    C(k) = norm(Pinv_sqrt * [eye(nx), K'] * Lmat(k,:)');
end
alpha = min(ones(nconstr,1)./C);
fprintf('\nTerminal alpha = %.6g\n', alpha);

%% ========================================================================
% (3) TIGHTENING LMIs (reduced): Pdelta_r, Kdelta_r
%% ========================================================================
con = [];

Xdr = sdpvar(nxr,nxr,'symmetric');   % Xdr = Pdelta_r^{-1}
Ydr = sdpvar(nu,nxr,'full');         % Ydr = Kdelta_r * Xdr

v0_worst = max(v_vertices);
Br_worst = [v0_worst/Lwb, 0;
            0,            1];
dwr = (Br_worst*uw')';

for i=1:Nv
    v0 = v_vertices(i);
    Adr = zeros(nxr,nxr);
    Bdr = [v0/Lwb, 0;
           0,      1];
    AXBY = Adr*Xdr + Bdr*Ydr;
    con = [con; (AXBY + AXBY' + 2*rho_c*Xdr) <= 0];
end

% Pdelta_r >= Pr  <=>  Xdr <= Xr_num
con = [con; Xdr <= Xr_num];

% input-compatibility (conservative)
for j=1:nconstr
    ineq = [1,               Lu(j,:)*Ydr; ...
            (Lu(j,:)*Ydr)',  Xdr         ];
    con = [con; ineq >= 0];
end

% disturbance energy bound
for j=1:size(dwr,1)
    ineq = [Xdr,       dwr(j,:)'; ...
            dwr(j,:),  wbar^2    ];
    con = [con; ineq >= 0];
end

disp("Solving TIGHTENING LMIs for Pdelta/Kdelta (reduced [psi v]) ...");
obj = trace(Xdr);              % small Xdr -> big Pdelta -> smaller alpha_s / smaller gamma_pos*s
sol2 = optimize(con, obj, ops);
fprintf('\n[Tightening LMI] sol.problem=%d (%s)\n', sol2.problem, yalmiperror(sol2.problem));
if sol2.problem ~= 0
    error("Tightening LMI infeasible/failed.");
end

Xdr_num = value(Xdr);
Ydr_num = value(Ydr);

Pdelta_r = inv(Xdr_num);
Kdelta_r = Ydr_num * Pdelta_r;

Pdelta = blkdiag(Ppos_delta, Pdelta_r);
Kdelta = zeros(nu,nx);
Kdelta(:,3:4) = Kdelta_r;

cj = zeros(nconstr,1);
Pdelta_inv_sqrt = inv(sqrtm(Pdelta));
for i=1:nconstr
    cj(i) = norm(Pdelta_inv_sqrt * [eye(nx), Kdelta'] * [zeros(1,nx), Lu(i,:)]');
end

alpha_s = norm(sqrtm(P) * inv(sqrtm(Pdelta)));
fprintf('\nalpha_s = %.6g\n', alpha_s);

%% ========================================================================
% (4) OBSTACLE TIGHTENING COEFFICIENT gamma_pos
%     Use in NMPC: r_eff(k) = r_safe + gamma_pos * s(k)
%% ========================================================================
Spos = [1 0 0 0;
        0 1 0 0];

Mpos = Spos * (Pdelta \ eye(nx)) * Spos';
Mpos = (Mpos + Mpos')/2;
gamma_pos = sqrt(max(eig(Mpos)));

fprintf('\nObstacle tightening: gamma_pos = %.6g [m per unit s]\n', gamma_pos);
fprintf('Use in NMPC: r_eff(k) = r_safe + gamma_pos*s(k)\n');

% Worst-case s over horizon (sdot = -rho*s + wbar, s(0)=0), rho=rho_c
s_max = (1 - exp(-rho_c * Tf)) / rho_c * wbar;
r_tube_max = gamma_pos * s_max;
fprintf('Optional: s_max(Tf)=%.6g, r_tube_max=%.6g [m]\n', s_max, r_tube_max);

%% ========================================================================
% (5) Pack outputs (values that normally go to files)
%% ========================================================================
OUT = struct();
OUT.P        = P;
OUT.Q        = Q;
OUT.R        = R;
OUT.K        = K;
OUT.alpha    = alpha;
OUT.Kinit    = Kinit;
OUT.wbar     = wbar;
OUT.rho_c    = rho_c;
OUT.Pdelta   = Pdelta;
OUT.Kdelta   = Kdelta;
OUT.Lx       = Lx;
OUT.Lu       = Lu;
OUT.Ls       = cj;           % tightening coefficients (Ls in python)
OUT.Tf       = Tf;
OUT.alpha_s  = alpha_s;
OUT.gamma_pos = gamma_pos;
OUT.r_tube_max = r_tube_max;

%% ========================================================================
% EXPORT or PRINT
%% ========================================================================
if writeout
    if ~exist(outdir,"dir"), mkdir(outdir); end

    writematrix(reshape(round(P,6),1,[]),          fullfile(outdir,'P.txt'));
    writematrix(reshape(round(Q,6),1,[]),          fullfile(outdir,'Q.txt'));
    writematrix(reshape(round(R,6),1,[]),          fullfile(outdir,'R.txt'));
    writematrix(reshape(round(K,6),1,[]),          fullfile(outdir,'K.txt'));
    writematrix(alpha,                            fullfile(outdir,'alpha.txt'));
    writematrix(reshape(round(Kinit,6),1,[]),      fullfile(outdir,'Kinit.txt'));

    writematrix(round(wbar,6),                     fullfile(outdir,'wbar.txt'));
    writematrix(round(rho_c,6),                    fullfile(outdir,'rho_c.txt'));
    writematrix(reshape(round(Pdelta,6),1,[]),     fullfile(outdir,'Pdelta.txt'));
    writematrix(reshape(round(Kdelta,6),1,[]),     fullfile(outdir,'Kdelta.txt'));

    writematrix(reshape(round(Lx',6),1,[]),        fullfile(outdir,'Lx.txt'));
    writematrix(reshape(round(Lu',6),1,[]),        fullfile(outdir,'Lu.txt'));
    writematrix(reshape(round(cj,6),1,[]),         fullfile(outdir,'Ls.txt'));
    writematrix(Tf,                               fullfile(outdir,'Tf.txt'));
    writematrix(alpha_s,                          fullfile(outdir,'alpha_s.txt'));
    writematrix(gamma_pos,                        fullfile(outdir,'gamma_pos.txt'));
    writematrix(r_tube_max,                       fullfile(outdir,'r_tube_max.txt'));

    fprintf("\nExported to %s/\n", outdir);
else
    fprintf("\n================ WOULD EXPORT THESE VALUES ================\n");

    fprintf("\nP (4x4):\n"); disp(OUT.P);
    fprintf("Q (4x4):\n"); disp(OUT.Q);
    fprintf("R (2x2):\n"); disp(OUT.R);
    fprintf("K (2x4):\n"); disp(OUT.K);
    fprintf("alpha:\n"); disp(OUT.alpha);
    fprintf("Kinit (2x4):\n"); disp(OUT.Kinit);

    fprintf("wbar:\n"); disp(OUT.wbar);
    fprintf("rho_c:\n"); disp(OUT.rho_c);
    fprintf("Pdelta (4x4):\n"); disp(OUT.Pdelta);
    fprintf("Kdelta (2x4):\n"); disp(OUT.Kdelta);

    fprintf("Lx (4x4):\n"); disp(OUT.Lx);
    fprintf("Lu (4x2):\n"); disp(OUT.Lu);
    fprintf("Ls=cj (4x1):\n"); disp(OUT.Ls);

    fprintf("Tf:\n"); disp(OUT.Tf);
    fprintf("alpha_s:\n"); disp(OUT.alpha_s);
    fprintf("gamma_pos:\n"); disp(OUT.gamma_pos);
    fprintf("r_tube_max:\n"); disp(OUT.r_tube_max);

    fprintf("\n=== reshape() vectors exactly like your .txt convention ===\n");
    fprintf("P.txt row:\n");     disp(reshape(round(OUT.P,6),1,[]));
    fprintf("Q.txt row:\n");     disp(reshape(round(OUT.Q,6),1,[]));
    fprintf("R.txt row:\n");     disp(reshape(round(OUT.R,6),1,[]));
    fprintf("K.txt row:\n");     disp(reshape(round(OUT.K,6),1,[]));
    fprintf("Kinit.txt row:\n"); disp(reshape(round(OUT.Kinit,6),1,[]));
    fprintf("Pdelta.txt row:\n");disp(reshape(round(OUT.Pdelta,6),1,[]));
    fprintf("Kdelta.txt row:\n");disp(reshape(round(OUT.Kdelta,6),1,[]));
    fprintf("Lx.txt row (transpose then reshape):\n"); disp(reshape(round(OUT.Lx',6),1,[]));
    fprintf("Lu.txt row (transpose then reshape):\n"); disp(reshape(round(OUT.Lu',6),1,[]));
    fprintf("Ls.txt row (cj):\n"); disp(reshape(round(OUT.Ls,6),1,[]));
    fprintf("alpha.txt:\n"); disp(OUT.alpha);
    fprintf("Tf.txt:\n"); disp(OUT.Tf);
    fprintf("alpha_s.txt:\n"); disp(OUT.alpha_s);
    fprintf("gamma_pos.txt:\n"); disp(OUT.gamma_pos);
    fprintf("r_tube_max.txt:\n"); disp(OUT.r_tube_max);

    fprintf("==========================================================\n");
end

%% ========================================================================
% TESTS / PLOTS: compare controllers Kinit / K / Kdelta
%% ========================================================================
fprintf('\n================ CONTROLLER COMPARISON TESTS ================\n');

sim_reduced = @(Kr_use, v0, xr0, Ts, Tend) local_sim_reduced(Kr_use, v0, xr0, Ts, Tend, Lwb);

v_test = 10.0;
xr0_list = [ 0.30  8.0;
            -0.25  6.0;
             0.15 -2.0]';

Ts = 0.01; Tend = 5.0;

Kr_init = Kinit(:,3:4);
Kr_lmi  = K(:,3:4);
Kr_del  = Kdelta(:,3:4);

labels = ["Kinit (LQR warmstart)", "K (LMI terminal)", "Kdelta (tightening)"];
Krs    = {Kr_init, Kr_lmi, Kr_del};

if PLOT_FIGURES
    figure(1); clf; grid on; hold on;
    title('Reduced response: psi(t) comparison'); xlabel('t (s)'); ylabel('psi (rad)');

    figure(2); clf; grid on; hold on;
    title('Reduced response: v(t) comparison'); xlabel('t (s)'); ylabel('v (m/s)');

    figure(3); clf; grid on; hold on;
    title('Control comparison: delta(t) (unsat, reduced model)'); xlabel('t (s)'); ylabel('delta (rad)');

    figure(4); clf; grid on; hold on;
    title('Control comparison: a(t) (unsat, reduced model)'); xlabel('t (s)'); ylabel('a (m/s^2)');
end

for ic = 1:size(xr0_list,2)
    xr0 = xr0_list(:,ic);
    for j=1:numel(Krs)
        Kr_use = Krs{j};
        [t, xr_hist, u_hist] = sim_reduced(Kr_use, v_test, xr0, Ts, Tend);

        if PLOT_FIGURES
            figure(1); plot(t, xr_hist(1,:), 'DisplayName', sprintf('%s ic%d', labels(j), ic));
            figure(2); plot(t, xr_hist(2,:), 'DisplayName', sprintf('%s ic%d', labels(j), ic));
            t_u = t(1:end-1);
            figure(3); plot(t_u, u_hist(1,:), 'DisplayName', sprintf('%s ic%d', labels(j), ic));
            figure(4); plot(t_u, u_hist(2,:), 'DisplayName', sprintf('%s ic%d', labels(j), ic));
        end
    end
end

if PLOT_FIGURES
    figure(1); legend('Location','best');
    figure(2); legend('Location','best');
    figure(3); legend('Location','best');
    figure(4); legend('Location','best');
end

%% ========================================================================
% Nonlinear quick sim comparing Kinit vs K (with saturation)
%% ========================================================================
if RUN_FULL_NL_SIM
    fprintf('\n[NL] Nonlinear sanity compare Kinit vs K (tan + sat)\n');

    x0 = [0; 0; 0.25; 8.0];
    Ts_nl = 0.01; Tend_nl = 6.0;

    [t1, x1, u1] = local_sim_nonlinear(Kinit, x0, Ts_nl, Tend_nl, Lwb, delta_max, a_min, a_max, uw, USE_INPUT_DIST);
    [t2, x2, u2] = local_sim_nonlinear(K,     x0, Ts_nl, Tend_nl, Lwb, delta_max, a_min, a_max, uw, USE_INPUT_DIST);

    if PLOT_FIGURES
        figure; grid on; hold on;
        plot(t1, x1(3,:), 'DisplayName','psi Kinit');
        plot(t2, x2(3,:), 'DisplayName','psi K');
        xlabel('t (s)'); ylabel('psi (rad)'); title('Nonlinear psi(t): Kinit vs K');
        legend('Location','best');

        figure; grid on; hold on;
        plot(t1, x1(4,:), 'DisplayName','v Kinit');
        plot(t2, x2(4,:), 'DisplayName','v K');
        xlabel('t (s)'); ylabel('v (m/s)'); title('Nonlinear v(t): Kinit vs K');
        legend('Location','best');

        figure; grid on; hold on;
        stairs(t1(1:end-1), u1(1,:), 'DisplayName','delta Kinit');
        stairs(t2(1:end-1), u2(1,:), 'DisplayName','delta K');
        yline(delta_max,'--'); yline(-delta_max,'--');
        xlabel('t (s)'); ylabel('delta (rad)'); title('Nonlinear delta(t) with saturation');
        legend('Location','best');

        figure; grid on; hold on;
        stairs(t1(1:end-1), u1(2,:), 'DisplayName','a Kinit');
        stairs(t2(1:end-1), u2(2,:), 'DisplayName','a K');
        yline(a_max,'--'); yline(a_min,'--');
        xlabel('t (s)'); ylabel('a (m/s^2)'); title('Nonlinear a(t) with saturation');
        legend('Location','best');
    end
end

fprintf('\n================ DONE ================\n');

%% ========================================================================
% LOCAL FUNCTIONS
%% ========================================================================
function [t, xr_hist, u_hist] = local_sim_reduced(Kr, v0, xr0, Ts, Tend, Lwb)
    N = round(Tend/Ts);
    t = (0:N)*Ts;

    xr_hist = zeros(2, N+1);
    u_hist  = zeros(2, N);

    xr_hist(:,1) = xr0;

    Br = [v0/Lwb, 0;
          0,      1];
    Ar = zeros(2,2);

    for k=1:N
        u = Kr * xr_hist(:,k);
        u_hist(:,k) = u;
        xdot = Ar*xr_hist(:,k) + Br*u;
        xr_hist(:,k+1) = xr_hist(:,k) + Ts*xdot;
    end
end

function [t, x, u_hist] = local_sim_nonlinear(Kfull, x0, Ts, Tend, Lwb, delta_max, a_min, a_max, uw, use_dist)
    N = round(Tend/Ts);
    t = (0:N)*Ts;

    x = zeros(4, N+1);
    u_hist = zeros(2, N);
    x(:,1) = x0;

    for k=1:N
        u = Kfull * x(:,k);

        if use_dist
            idx = randi(size(uw,1));
            u = u + uw(idx,:)';
        end

        u(1) = min(max(u(1), -delta_max), delta_max);
        u(2) = min(max(u(2),  a_min),     a_max);

        u_hist(:,k) = u;

        psi = x(3,k);
        v   = x(4,k);
        delta = u(1);
        a     = u(2);

        xdot = [ v*cos(psi);
                 v*sin(psi);
                 v/Lwb * tan(delta);
                 a ];

        x(:,k+1) = x(:,k) + Ts*xdot;
    end
end
