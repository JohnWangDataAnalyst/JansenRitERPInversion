"""

WhoBPyT Jansen-Rit model classes
---------------------------------

Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak

Neural Mass Model fitting module for JR with connections from pyramidal to pyramidal, excitatory, and inhibitory populations for M/EEG

"""


"""
Importage
---------
"""

# PyTorch stuff
from torch.nn.parameter import Parameter as ptParameter
from torch.nn import ReLU as ptReLU
from torch.linalg import norm as ptnorm
from torch import (tensor as pttensor, float32 as ptfloat32, sum as ptsum, exp as ptexp, diag as ptdiag, 
                   transpose as pttranspose, zeros_like as ptzeros_like, int64 as ptint64, randn as ptrandn, 
                   matmul as ptmatmul, tanh as pttanh, matmul as ptmatmul, reshape as ptreshape, sqrt as ptsqrt,
                   ones as ptones, cat as ptcat)

# Numpy stuff
from numpy.random import uniform 
from numpy import ones,zeros

# WhoBPyT stuff
from ...datatypes import AbstractNeuralModel, AbstractParams, Parameter as par
from ...functions.arg_type_check import method_arg_type_check



"""
JR params class
---------------
"""

class RNNJANSEN(AbstractNMM):
    """
    A module for forward model (JansenRit) to simulate EEG signals

    Attibutes
    ---------
    state_size : int
        Number of states in the JansenRit model

    output_size : int
        Number of EEG channels.

    node_size: int
        Number of ROIs

    hidden_size: int
        Number of step_size for each sampling step

    step_size: float
        Integration step for forward model

    tr : float # TODO: CHANGE THE NAME TO sampling_rate
        Sampling rate of the simulated EEG signals

    TRs_per_window: int # TODO: CHANGE THE NAME
        Number of EEG signals to simulate

    sc: ndarray (node_size x node_size) of floats
        Structural connectivity

    lm: ndarray of floats
        Leadfield matrix from source space to EEG space

    dist: ndarray of floats
        Distance matrix

    use_fit_gains: bool
        Flag for fitting gains. 1: fit, 0: not fit

    use_fit_lfm: bool
        Flag for fitting the leadfield matrix. 1: fit, 0: not fit

    # FIGURE OUT: g, c1, c2, c3, c4: tensor with gradient on
    #     model parameters to be fit

    std_in: tensor with gradient on
        Standard deviation for input noise

    params: ParamsJR
        Model parameters object.


    Methods
    -------
    createIC(self, ver):
        Creates the initial conditions for the model.

    createDelayIC(self, ver):
        Creates the initial conditions for the delays.

    setModelParameters(self):
        Sets the parameters of the model.

    forward(input, noise_out, hx)
        Forward pass for generating a number of EEG signals with current model parameters

    """

    def __init__(self, params: ParamsJR, node_size=200,
                 TRs_per_window=20, step_size=0.0001, output_size= 62, tr=0.001, sc=np.ones((200,200)), lm=np.ones((62,200)), dist=np.ones((200,200)), use_fit_gains=True, mask = np.ones((200,200))):
        """
        Parameters
        ----------
        node_size: int
            Number of ROIs
        TRs_per_window: int # TODO: CHANGE THE NAME
            Number of EEG signals to simulate
        step_size: float
            Integration step for forward model
        output_size : int
            Number of EEG channels.
        tr : float # TODO: CHANGE THE NAME TO sampling_rate
            Sampling rate of the simulated EEG signals
        sc: ndarray node_size x node_size float array
            Structural connectivity
        lm: ndarray float array
            Leadfield matrix from source space to EEG space
        dist: ndarray float array
            Distance matrix
        use_fit_gains: bool
            Flag for fitting gains. 1: fit, 0: not fit
        params: ParamsJR
            Model parameters object.
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types

        super(RNNJANSEN, self).__init__(params)

        self.pop_names = np.array(['P', 'E', 'I'])
        self.state_names = np.array(['current', 'voltage'])
        self.output_names = ["eeg"]
        self.track_params = [] #Is populated during setModelParameters()

        self.model_name = "JR"
        self.pop_size = 3  # 3 populations JR
        self.state_size = 2  # 2 states in each population
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32)  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of EEG channels
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.dist = torch.tensor(dist, dtype=torch.float32)
        self.lm = lm
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        #self.use_fit_lfm = use_fit_lfm
        self.params = params
        self.output_size = lm.shape[0]  # number of EEG channels
        self.mask = mask

        self.setModelParameters()
        #self.setModelSCParameters()

    def createIC(self, ver):
        """
        Creates the initial conditions for the model.

        Parameters
        ----------
        ver : int # TODO: ADD MORE DESCRIPTION
            Initial condition version. (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, state_size) with random values between `state_lb` and `state_ub`.
        """

        state_lb = -0.01
        state_ub = 0.01

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size, self.pop_size, self.state_size)),
                             dtype=torch.float32)

    def createDelayIC(self, ver):
        """
        Creates the initial conditions for the delays.

        Parameters
        ----------
        ver : int
            Initial condition version. (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random values between `state_lb` and `state_ub`.
        """

        delays_max = 500
        state_ub = 0.1
        state_lb = 0

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size,  delays_max)), dtype=torch.float32)

    def setModelSCParameters(self):
        """
        Sets the parameters of the model.
        """
         # Create the arrays in numpy
        small_constant = 0.05
        n_nodes = self.node_size
        zsmat = zeros((self.node_size, 2)) + small_constant
        w_p2e = zsmat.copy() # the pyramidal to excitatory interneuron cross-layer gains
        w_p2i = zsmat.copy() # the pyramidal to inhibitory interneuron cross-layer gains
        w_p2p = zsmat.copy() # the pyramidal to pyramidal cells same-layer gains

        # Set w_bb, w_ff, and w_ll as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            self.w_bb = ptParameter(pttensor(w_p2i, dtype=ptfloat32))
            self.w_ff = ptParameter(pttensor(w_p2e, dtype=ptfloat32))
            self.w_ll = ptParameter(pttensor(w_p2p, dtype=ptfloat32))
            self.params_fitted['modelparameter'].append(self.w_ll)
            self.params_fitted['modelparameter'].append(self.w_ff)
            self.params_fitted['modelparameter'].append(self.w_bb)
        else:
            self.w_bb = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
            self.w_ff = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
            self.w_ll = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)


    def forward(self, external, hx, hE):
        """
        This function carries out the forward Euler integration method for the JR neural mass model,
        with time delays, connection gains, and external inputs considered. Each population (pyramidal,
        excitatory, inhibitory) in the network is modeled as a non-linear second order system. The function
        updates the state of each neural population and computes the EEG signals at each time step.

        Parameters
        ----------
        external : torch.Tensor
            Input tensor of shape (batch_size, num_ROIs) representing the input to the model.
        hx : Optional[torch.Tensor]
            Optional tensor of shape (batch_size, state_size, num_ROIs) representing the initial hidden state.
        hE : Optional[torch.Tensor]
            Optional tensor of shape (batch_size, num_ROIs, delays_max) representing the initial delays.

        Returns
        -------
        next_state : dict
            Dictionary containing the updated current state, EEG signals, and the history of
            each population's current and voltage at each time step.

        hE : torch.Tensor
            Tensor representing the updated history of the pyramidal population's current.
        """

        # Generate the ReLU module
        m = torch.nn.ReLU()

        # Define some constants
        con_1 = torch.tensor(1.0, dtype=torch.float32) # Define constant 1 tensor
        conduct_lb = 0  # lower bound for conduct velocity
        u_2ndsys_ub = 800  # the bound of the input for second order system
        noise_std_lb = 0  # lower bound of std of noise
        lb = 0.  # lower bound of local gains
        k_lb = 0.5  # lower bound of coefficient of external inputs


        # Defining NMM Parameters to simplify later equations
        #TODO: Change code so that params returns actual value used without extras below
        A = 0 * con_1 + m(self.params.A.value())
        a = 0 * con_1 + m(self.params.a.value())
        B = 0 * con_1 + m(self.params.B.value())
        b = 0 * con_1 + m(self.params.b.value())
        g = 1*(lb * con_1 + m(self.params.g.value()))
        c1 = (lb * con_1 + m(self.params.c1.value()))
        c2 = (lb * con_1 + m(self.params.c2.value()))
        c3 = (lb * con_1 + m(self.params.c3.value()))
        c4 = (lb * con_1 + m(self.params.c4.value()))
        std_in = (noise_std_lb * con_1 + m(self.params.std_in.value())) #around 20
        vmax = self.params.vmax.value()
        v0 = self.params.v0.value()
        r = self.params.r.value()
        y0 = self.params.y0.value()
        mu = (0.1 * con_1 + m(self.params.mu.value()))
        k = (5.0 * con_1 + m(self.params.k.value()))
        cy0 = self.params.cy0.value()
        ki = self.params.ki.value()

        g_f = 1*(lb * con_1 + m(self.params.g_f.value()))
        g_b = 1*(lb * con_1 + m(self.params.g_b.value()))
        w_bb = 1*(lb * con_1 + m(self.params.w_bb.value()))
        w_ff = 1*(lb * con_1 + m(self.params.w_ff.value()))
        w_ll = 1*(lb * con_1 + m(self.params.w_ll.value()))
        lm = self.params.lm.value()

        next_state = {}

        P = hx[:, 0:1, 0]  # current of pyramidal population
        E = hx[:, 1:2, 0]  # current of excitory population
        I = hx[:, 2:3, 0]  # current of inhibitory population

        Pv = hx[:, 0:1, 1]  # voltage of pyramidal population
        Ev = hx[:, 1:2, 1]  # voltage of exictory population
        Iv = hx[:, 2:3, 1]  # voltage of inhibitory population
        #print(M.shape)
        dt = self.step_size
        n_nodes = self.node_size
        n_chans = self.output_size

        sc = self.sc
        ptsc = pttensor(sc, dtype=ptfloat32)

        if self.sc.shape[0] > 1:

            # Update the Laplacian based on the updated connection gains w_bb.
            #w_b = ptdiag(ptexp(self.w_bb[:,0])) @ ptsc @ ptdiag(ptexp(self.w_bb[:,1]))
            w_b = ptdiag(w_bb[:,0]) @ ptsc @ ptdiag(w_bb[:,1])
            w_n_b = w_b #/ ptnorm(w_b)*pttensor(self.mask, dtype=ptfloat32)
            self.sc_m_b = w_n_b
            dg_b = -ptdiag(ptsum(w_n_b, dim=1))

            # Update the Laplacian based on the updated connection gains w_ff.
            w_f = ptdiag(w_ff[:,0]) @ ptsc @ ptdiag(w_ff[:,1])
            #w_f = ptdiag(ptexp(self.w_ff[:,0])) @ ptsc @ ptdiag(ptexp(self.w_ff[:,1]))
            w_n_f = w_f #/ ptnorm(w_f)*pttensor(self.mask, dtype=ptfloat32)
            self.sc_m_f = w_n_f
            dg_f = -ptdiag(ptsum(w_n_f, dim=1))

            # Update the Laplacian based on the updated connection gains w_ll.
            w_l = ptdiag(w_ll[:,0]) @ ptsc @ ptdiag(w_ll[:,0])
            #w_l =ptexp(self.w_ll) * ptsc
            w_n_l = (0.5 * (w_l + pttranspose(w_l, 0, 1))) #/ ptnorm(0.5 * (w_l + pttranspose(w_l, 0, 1)))*pttensor(self.mask, dtype=ptfloat32)
            self.sc_fitted = w_n_l
            dg_l = -ptdiag(ptsum(w_n_l, dim=1))
        else:
            l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32) #TODO: This is not being called anywhere
            dg_l = 0
            dg_b = 0
            dg_f = 0
            w_n_l = 0
            w_n_b = 0
            w_n_f = 0

        self.delays = (self.dist / mu).type(torch.int64)

        # Placeholder for the updated current state
        current_state = ptzeros_like(hx)

        # Initializing lists for the history of the EEG signals, as well as each population's current and voltage.
        eeg_window = []
        E_window = []
        I_window = []
        P_window = []
        Ev_window = []
        Iv_window = []
        Pv_window = []
        states_window = []
        inputs_window = []

        # Use the forward model to get EEG signal at the i-th element in the window.
        for i_window in range(self.TRs_per_window):
            Ed = pttranspose(hE.clone().gather(1,self.delays), 0, 1)

            LEd_p2i_adj = ptreshape(ptsum(w_n_b * Ed, 1), (n_nodes, 1)) #+ ptmatmul(dg_b, I -E)
            LEd_p2e_adj = ptreshape(ptsum(w_n_f * Ed, 1), (n_nodes, 1)) #+ ptmatmul(dg_f, E -I)
            LEd_p2p_adj = ptreshape(ptsum(w_n_l * Ed, 1), (n_nodes, 1)) #+ ptmatmul(dg_l, P)
            for step_i in range(self.steps_per_TR):

                LEd_p2i = LEd_p2i_adj + ptmatmul(dg_b, I)
                LEd_p2e = LEd_p2e_adj + ptmatmul(dg_f, E)
                LEd_p2p = LEd_p2p_adj + ptmatmul(dg_l, P)
                # external input
                u_stim = external[:, step_i:step_i + 1, i_window, 0]

                # Stochastic / noise term
                P_noise = std_in * ptrandn(n_nodes, 1)
                E_noise = std_in * ptrandn(n_nodes, 1)
                I_noise = std_in * ptrandn(n_nodes, 1)

                # Compute the firing rate for each neural populatin
                # at every node using the wave-to-pulse (sigmoid) functino
                # (vmax = max value of sigmoid, v0 = midpoint of sigmoid)
                P_sigm = vmax / ( 1 + ptexp ( r*(v0 -  (E-I) ) ) )
                E_sigm = vmax / ( 1 + ptexp ( r*(v0 - (c1*P) ) ) )
                I_sigm = vmax / ( 1 + ptexp ( r*(v0 - (c3*P) ) ) )
                # Sum the four different input types into a single input value for each neural
                # populatin state variable
                # The four input types are:
                # - Local      (L)      - from other neural populations within a node (E->P,P->I, etc.)
                # - Long-range (L-R)    - from other nodes in the network, weighted by the long-range
                #                         connectivity matrices, and time-delayed
                # - Noise      (N)      - stochastic noise input
                # - External   (E)      - external stimulation, eg from TMS or sensory stimulus
                #
                #        Local    Long-range   Noise   External

                rP =     P_sigm  + g*LEd_p2p   + P_noise + k*ki*u_stim
                rE =  c2*E_sigm  + g_f*LEd_p2e + E_noise
                rI =  c4*I_sigm  + g_b*LEd_p2i + I_noise

                # Apply some additional scaling
                rP_bd = u_2ndsys_ub * pttanh(rP / u_2ndsys_ub)
                rE_bd = u_2ndsys_ub * pttanh(rE / u_2ndsys_ub)
                rI_bd = u_2ndsys_ub * pttanh(rI / u_2ndsys_ub)

                # Compute d/dt   ('_tp1' = state variable at time t+1)
                P_tp1 =  P + dt * Pv
                E_tp1 =  E + dt * Ev
                I_tp1 =  I + dt * Iv
                Pv_tp1 = Pv + dt * ( A*a*rP_bd  -  2*a*Pv  -  a**2 * P )
                Ev_tp1 = Ev + dt * ( A*a*rE_bd  -  2*a*Ev  -  a**2 * E )
                Iv_tp1 = Iv + dt * ( B*b*rI_bd  -  2*b*Iv  -  b**2 * I )

                # Calculate the saturation for model states (for stability and gradient calculation).

                # Add some additional saturation on the model states
                # (for stability and gradient calculation).
                P = 1000*pttanh(P_tp1/1000)
                E = 1000*pttanh(E_tp1/1000)
                I = 1000*pttanh(I_tp1/1000)
                Pv = 1000*pttanh(Pv_tp1/1000)
                Ev = 1000*pttanh(Ev_tp1/1000)
                Iv = 1000*pttanh(Iv_tp1/1000)
                #print('after M', M.shape)
                # Update placeholders for pyramidal buffer
                #hE[:, 0] = P[:,0]

            # Capture the states at every tr in the placeholders for checking them visually.

            # Capture the states at every tr in the placeholders for checking them visually.
            hE = ptcat([P, hE[:, :-1]], dim=1)  # update placeholders for pyramidal buffer

            # Capture the states at every tr in the placeholders which is then used in the cost calculation.
            lm_t = (lm.T / torch.sqrt((lm ** 2).sum(1))).T
            lm_t_dm = (lm_t - 1 / n_chans * torch.matmul(torch.ones((1,n_chans)), lm_t))
            temp = cy0 * torch.matmul(lm_t_dm, (E-I)[:200]) - 1 * y0
            eeg_window.append(temp)
            states_window.append(torch.cat([torch.cat([P, E, I], dim=1)[:,:,np.newaxis], \
                                   torch.cat([Pv, Ev, Iv], dim=1)[:,:,np.newaxis]], dim=2)[:,:,:,np.newaxis])
            inputs_window.append(torch.cat([g*LEd_p2p, g_f*LEd_p2e, g_b*LEd_p2i], dim=1)[:,:,np.newaxis])
        # Update the current state.
        self.lm_t = lm_t_dm

        current_state = torch.cat([torch.cat([P, E, I], dim=1)[:,:,np.newaxis], \
                                   torch.cat([Pv, Ev, Iv], dim=1)[:,:,np.newaxis]], dim=2)
        next_state['current_state'] = current_state
        next_state['eeg'] = torch.cat(eeg_window, dim=1)
        next_state['states'] = torch.cat(states_window, dim=3)
        next_state['inputs'] = torch.cat(inputs_window, dim=2)


        return next_state, hE
