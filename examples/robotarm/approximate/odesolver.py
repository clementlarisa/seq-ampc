import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import diffrax
import optimistix
# from diffrax import diffeqsolve, ODETerm, Dopri5

class FuncPrestabilized(eqx.Module):
    dt:  float
    Tf:  float
    nu:  int
    f:   callable
    N:   int
    ts:  jax.Array
    u_min: jax.Array
    u_max: jax.Array
    Kdelta: jax.Array

    def __init__(self, dt, nu, f, N, Kdelta, u_min, u_max):
        super().__init__()
        self.dt = dt 
        self.Tf = dt*N
        self.nu = nu 
        self.f = f 
        self.N = N
        ts = jnp.linspace(0, self.dt*self.N, self.N+1)
        self.ts=jnp.repeat(ts, 2, axis=0)[1:-1]
        self.Kdelta = jnp.array(Kdelta)
        self.u_min = jnp.array(u_min)
        self.u_max = jnp.array(u_max)

    def __call__(self,t,y,args):
        U = args
        # U = U.reshape((self.N, self.nu))
        U = jnp.repeat(U, 2, axis=0)

        U_piecewise_const_interp = diffrax.LinearInterpolation(self.ts, U)
        u = U_piecewise_const_interp.evaluate(t, left=False)
        
        u_stab = self.Kdelta@y + u
        u_out = jnp.maximum(jnp.minimum(u_stab, self.u_max), self.u_min)
        
        return self.f(y, u_out)
    
class FuncPwConstClip(eqx.Module):
    dt:  float
    Tf:  float
    nu:  int
    f:   callable
    N:   int
    ts:  jax.Array
    u_min: jax.Array
    u_max: jax.Array

    def __init__(self, dt, nu, f, N, u_min, u_max):
        super().__init__()
        self.dt = dt 
        self.Tf = dt*N
        self.nu = nu 
        self.f = f 
        self.N = N
        ts = jnp.linspace(0, self.dt*self.N, self.N+1)
        self.ts=jnp.repeat(ts, 2, axis=0)[1:-1]
        self.u_min = jnp.array(u_min)
        self.u_max = jnp.array(u_max)

    def __call__(self,t,y,args):
        U = args
        # U = U.reshape((self.N, self.nu))
        U = jnp.repeat(U, 2, axis=0)

        U_piecewise_const_interp = diffrax.LinearInterpolation(self.ts, U)
        u = U_piecewise_const_interp.evaluate(t, left=False)
        
        u_out = jnp.maximum(jnp.minimum(u, self.u_max), self.u_min)
        
        return self.f(y, u_out)
    
class FuncPwConst(eqx.Module):
    dt:  float
    Tf:  float
    nu:  int
    f:   callable
    N:   int
    ts:  jax.Array

    def __init__(self, dt, nu, f, N):
        super().__init__()
        self.dt = dt 
        self.Tf = dt*N
        self.nu = nu 
        self.f = f 
        self.N = N
        ts = jnp.linspace(0, self.dt*self.N, self.N+1)
        self.ts=jnp.repeat(ts, 2, axis=0)[1:-1]

    def __call__(self,t,y,args):
        U = args
        # U = U.reshape((self.N, self.nu))
        U = jnp.repeat(U, 2, axis=0)

        U_piecewise_const_interp = diffrax.LinearInterpolation(self.ts, U)
        u = U_piecewise_const_interp.evaluate(t, left=False)
        
        return self.f(y, u)

    
class IntegratorImplicitVariable(eqx.Module):
    N: int
    dt: float
    func: FuncPrestabilized
    nu: int
    root_finder: optimistix.AbstractRootFinder
    solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, func, nu):
        super().__init__()
        self.N = N
        self.dt = dt
        self.func = func
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        self.root_finder = optimistix.Newton(rtol=1.4e-8, atol=1.4e-8)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=100)

    def __call__(self, x, U):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=diffrax.ODETerm(self.func),
            # solver=diffrax.Dopri5(),
            # solver=diffrax.Kvaerno5(),
            solver = self.solver,
            t0 = 0, t1 = self.N*self.dt, dt0=self.dt,
            args=U,
            saveat=diffrax.SaveAt(ts=jax.numpy.linspace(0,self.N*self.dt, self.N+1)),
            # max_steps=100000,
            stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.4e-8, atol=1.4e-8),
            # stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys
class IntegratorImplicitFixed(eqx.Module):
    N: int
    dt: float
    func: FuncPrestabilized
    nu: int
    root_finder: optimistix.AbstractRootFinder
    solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, func, nu):
        super().__init__()
        self.N = N
        self.dt = dt
        self.func = func
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        self.root_finder = optimistix.Newton(rtol=1.4e-8, atol=1.4e-8)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=100)

    def __call__(self, x, U):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=diffrax.ODETerm(self.func),
            # solver=diffrax.Dopri5(),
            # solver=diffrax.Kvaerno5(),
            solver = self.solver,
            t0 = 0, t1 = self.N*self.dt, dt0=self.dt/10,
            args=U,
            saveat=diffrax.SaveAt(ts=jax.numpy.linspace(0,self.N*self.dt, self.N+1)),
            # max_steps=100000,
            # stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.4e-8, atol=1.4e-8),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys
class IntegratorImplicitEulerFixed(eqx.Module):
    N: int
    dt: float
    func: FuncPrestabilized
    nu: int
    root_finder: optimistix.AbstractRootFinder
    solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, func, nu):
        super().__init__()
        self.N = N
        self.dt = dt
        self.func = func
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        self.root_finder = optimistix.Newton(rtol=1.4e-8, atol=1.4e-8)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        self.solver = diffrax.ImplicitEuler(root_finder=self.root_finder, root_find_max_steps=100)

    def __call__(self, x, U):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=diffrax.ODETerm(self.func),
            # solver=diffrax.Dopri5(),
            # solver=diffrax.Kvaerno5(),
            solver = self.solver,
            t0 = 0, t1 = self.N*self.dt, dt0=self.dt,
            args=U,
            saveat=diffrax.SaveAt(ts=jax.numpy.linspace(0,self.N*self.dt, self.N+1)),
            # max_steps=100000,
            # stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.4e-8, atol=1.4e-8),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys
    
class IntegratorExplicitEulerFixed(eqx.Module):
    N: int
    dt: float
    func: FuncPrestabilized
    nu: int
    # solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, func, nu):
        super().__init__()
        self.N = N
        self.dt = dt
        self.func = func
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1.4e-8, atol=1.4e-8)
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=5)
        # self.solver = diffrax.Dopri5(),

    def __call__(self, x, U):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=diffrax.ODETerm(self.func),
            solver=diffrax.Euler(),
            # solver=diffrax.Kvaerno5(),
            # solver = self.solver,
            t0 = 0, t1 = self.N*self.dt, dt0=self.dt,
            args=U,
            saveat=diffrax.SaveAt(ts=jax.numpy.linspace(0,self.N*self.dt, self.N+1)),
            throw=False,
            # max_steps=100000,
            # stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.4e-8, atol=1.4e-8),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys
class IntegratorExplicitFixed(eqx.Module):
    N: int
    dt: float
    func: FuncPrestabilized
    nu: int
    # solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, func, nu):
        super().__init__()
        self.N = N
        self.dt = dt
        self.func = func
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1.4e-8, atol=1.4e-8)
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=5)
        # self.solver = diffrax.Dopri5(),

    def __call__(self, x, U):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=diffrax.ODETerm(self.func),
            solver=diffrax.Dopri5(),
            # solver=diffrax.Kvaerno5(),
            # solver = self.solver,
            t0 = 0, t1 = self.N*self.dt, dt0=self.dt/10,
            args=U,
            saveat=diffrax.SaveAt(ts=jax.numpy.linspace(0,self.N*self.dt, self.N+1)),
            throw=False,
            # max_steps=100000,
            # stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.4e-8, atol=1.4e-8),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys

    
class IntegratorExplicitVariable(eqx.Module):
    N: int
    dt: float
    func: FuncPrestabilized
    nu: int
    # solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, func, nu):
        super().__init__()
        self.N = N
        self.dt = dt
        self.func = func
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1.4e-8, atol=1.4e-8)
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=5)
        # self.solver = diffrax.Dopri5(),

    def __call__(self, x, U):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=diffrax.ODETerm(self.func),
            solver=diffrax.Dopri5(),
            # solver=diffrax.Kvaerno5(),
            # solver = self.solver,
            t0 = 0, t1 = self.N*self.dt, dt0=self.dt/10,
            args=U,
            saveat=diffrax.SaveAt(ts=jax.numpy.linspace(0,self.N*self.dt, self.N+1)),
            throw=False,
            # max_steps=100000,
            stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.4e-8, atol=1.4e-8),
            # stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys
    
    
class IntegratorShortHorizont(eqx.Module):
    N: int
    dt: float
    func: diffrax.ODETerm
    nu: int
    # solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, func, nu):
        super().__init__()
        self.N = N
        self.dt = dt
        self.func = diffrax.ODETerm(func)
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1.4e-8, atol=1.4e-8)
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=5)
        # self.solver = diffrax.Dopri5(),

    def __call__(self, x, U, t0):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=self.func ,
            solver=diffrax.Dopri5(),
            # solver=diffrax.Kvaerno5(),
            # solver = self.solver,
            t0 = t0, t1 = t0+self.dt, dt0=self.dt/10,
            args=U,
            # saveat=diffrax.SaveAt(ts=jax.numpy.linspace(0,self.N*self.dt, self.N+1)),
            throw=False,
            # max_steps=100000,
            # stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1.4e-8, atol=1.4e-8),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys[0]
    
class IntegratorImplicitShortHorizon(eqx.Module):
    N: int
    dt: float
    func: callable
    nu: int
    root_finder: optimistix.AbstractRootFinder
    solver: diffrax.AbstractSolver
    
    def __init__(self, N, dt, continuous_time_dynamics_func, nu, u_max, u_min):
        super().__init__()
        self.N = N
        self.dt = dt
        def func(t,y,args):
            u = args
            u_out = jnp.maximum(jnp.minimum(u, u_max), u_min)
            return continuous_time_dynamics_func(y, u_out)
        
        self.func = func
        self.nu = nu
        # self.root_finder = optimistix.Newton(rtol=1e-5, atol=1e-5)
        self.root_finder = optimistix.Newton(rtol=1e-14, atol=1e-14)
        self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=10000)
        # self.solver = diffrax.Kvaerno5(root_finder=self.root_finder, root_find_max_steps=100)

    def __call__(self, x, u):
        solution = diffrax.diffeqsolve(
            y0=x,
            terms=diffrax.ODETerm(self.func),
            # solver=diffrax.Dopri8(),
            # solver=diffrax.Kvaerno5(),
            solver = self.solver,
            t0 = 0, t1 = self.dt, dt0=self.dt,
            args=u,
            # max_steps=100000,
            # stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-12, atol=1e-12),
            stepsize_controller=diffrax.PIDController(rtol=1e-14, atol=1e-14),
            # stepsize_controller=diffrax.ConstantStepSize(),
        )
        return solution.ys[0]
class ScanIntegrator(eqx.Module):
    integrator : eqx.Module 
    idxs : jax.numpy.array
    
    def __init__(self, N, dt, continuous_time_dynamics_func, nu, batch_size, u_max, u_min):
        self.integrator = IntegratorImplicitShortHorizon(N=1, dt=dt, continuous_time_dynamics_func=continuous_time_dynamics_func, nu=nu, u_max=u_max, u_min=u_min)
        # self.idxs = jax.numpy.array(np.repeat([np.arange(N)], batch_size, axis=0).transpose())
        self.idxs = jax.numpy.array(np.arange(N))
        
    
    def __call__(self, X_, U_):
        def integrator_step(carry, idx):
            x, U = carry
            next_x = jax.vmap(self.integrator)(x, U[:,idx,:])
            carry = next_x, U
            return carry, next_x
        
        _, X_pred_ = jax.lax.scan(integrator_step, (X_[:,0,:], U_), self.idxs)
        X_pred_ = jax.numpy.moveaxis(X_pred_, 0, 1)
        X_pred_ = jax.numpy.insert(X_pred_, 0, X_[:,0,:], axis=1)
        return X_pred_
    

Integrator = IntegratorExplicitFixed
