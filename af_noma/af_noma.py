import numpy as np
from scipy.special import gamma
from scipy.special import binom
from scipy.special import erf
import matplotlib.pyplot as plt
from .multiFoxH import compMultiFoxH, compMultiFoxHIntegrand
from scipy.stats.sampling import NumericalInversePolynomial
import math
from itertools import product
import sympy

class AFFading:

    def __init__(self, alpha: float, ms: float, mu: float, z: float, gamma_b=1) -> None:
        
        self.alpha = alpha
        self.z = z
        self.ms = ms
        self.mu = mu
        self.psi = mu / ( ms - 1 )
        self.gamma_b = gamma_b
        self.lambda_p = (( 1.0 / self.psi )**( 2.0 / alpha )) * ( gamma( mu + 2.0/alpha ) * gamma( ms - 2.0 / alpha ) ) / ( gamma( mu ) * gamma( ms ) )

    def pdf( self, x ):

        if ( x < 1e-50 ) or ( math.isinf( x ) ):
            fdp_x = 0.0
        else:

            c1 = ( self.z**2 ) / ( 2 * x * gamma( self.mu ) * gamma( self.ms ) )
            csi_arg = ( self.psi * ( self.z**(self.alpha) * (self.lambda_p)**(self.alpha/2) ) * ( x**(self.alpha/2) ) ) / ( ( np.sqrt( self.gamma_b * ( self.z**2 + 2 ) ) )**self.alpha )
            a1 = 1 - self.ms
            a2 = ((self.z**2)/self.alpha) + 1
            b1 = self.mu
            b2 = ((self.z**2)/self.alpha)

            fdp_x = c1 * sympy.meijerg( [[a1],[a2]], [[b1,b2],[]], csi_arg ).evalf()

        return fdp_x

class NomaSystem:

    def __init__( self, num_users: int, tx_power: float, power_alloc: list, t_symbols_id: list, r_symbols_id: list, qam_order: int ) -> None:
        
        self.num_users = num_users
        self.power_alloc = np.array( power_alloc )
        self.tx_power = tx_power
        self.qam_order = qam_order
        self.num_bits_per_symbol = np.log2( qam_order ).astype(int)
        self.t_symbols_id = t_symbols_id
        self.r_symbols_id = r_symbols_id
        self.constellation = self.get_qam_const()
        self.gray_constellation = self.get_gray_qam_const()
        self.t_symbols = self.constellation[ t_symbols_id ]
        self.r_symbols = self.constellation[ r_symbols_id ]
        self.delta_l = np.array( self.t_symbols - self.r_symbols )

    def get_qam_const( self ):

        ref_const = np.zeros( self.qam_order, dtype=np.complex128 )
        # BPSK
        if self.qam_order == 2:

            ref_const[0] = -1.0
            ref_const[1] = 1.0
        # Reference QAM constellation
        else:
            # Squared order
            sq_qam_order = np.sqrt( self.qam_order ).astype(int)
            # Avg bit energy
            avg_bit_enrg = ( ( self.qam_order - 1 ) * 4 / 6 ) / ( np.log2( self.qam_order ) )
            k = 0
            for i in range(1, sq_qam_order + 1):
                for j in range(1, sq_qam_order + 1):

                    a_i = ( 2 * i - sq_qam_order - 1 ) / avg_bit_enrg
                    b_j = ( 2 * j - sq_qam_order - 1 ) / avg_bit_enrg
                    ref_const[ k ] = a_i + b_j*1j
                    k = k+1

        return ref_const

    def get_gray_qam_const( self ):

        # Squared order
        sq_qam_order = np.sqrt( self.qam_order ).astype(int)
        # Avg bit energy
        avg_bit_enrg = ( ( self.qam_order - 1 ) * 4 / 6 ) / ( np.log2( self.qam_order ) )
        # Reference Gray-coded QAM constellation
        ref_const_gray = np.zeros( self.qam_order, dtype=np.complex128 )
        k = 0
        real_p_qam = ( 2 * np.arange(1,sq_qam_order+1) - sq_qam_order - 1 ) / avg_bit_enrg
        imag_p_qam = ( 2 * np.arange(1,sq_qam_order+1) - sq_qam_order - 1 ) / avg_bit_enrg

        for i in range(0, sq_qam_order):
            for j in range(0, sq_qam_order):

                i_g = np.bitwise_xor(i,np.floor(i/2).astype(int))
                j_g = np.bitwise_xor(j,np.floor(j/2).astype(int))
                real_p = real_p_qam[ i_g ]
                imag_p = imag_p_qam[ j_g ]
                ref_const_gray[ k ] = real_p + imag_p*1j
                k = k+1

        return ref_const_gray

    def compute_theta_lth( self, noise_power: float, l: int ):

        # v_l term
        v_l = ( np.sqrt( 2 * noise_power ) * abs( self.delta_l[ l ] ) )
        # Signal term
        x_p = np.sqrt( self.power_alloc[ l ] * self.tx_power ) * ( np.abs( self.delta_l[ l ] ) )**2
        # Interference term
        iui = np.sum( np.sqrt( self.power_alloc[ l + 1 : self.num_users ] * self.tx_power ) * np.conj( self.t_symbols[ l + 1 : self.num_users ] ) )
        # SIC term
        sic = np.sum( np.sqrt( self.power_alloc[ 0 : l ] ) * np.conj( self.delta_l[ 0 : l ] ) )

        theta_l = (x_p / v_l) + ( 2 / v_l ) * np.real( self.delta_l[ l ] * ( iui + sic ) )

        return theta_l

    def compute_theta_l( self, noise_power: float ):

        # Theta vector
        theta_l = np.zeros( self.num_users )
        for l in range(0, self.num_users):

            theta_l[ l ] = self.compute_theta_lth( noise_power, l )

        return theta_l


def get_lth_pep_hfox_tuples( l: int, k: int, fp: AFFading ):

    # mn-tuples
    mn_t = [(0,2)] + [(2,1)] + [(2,2)] * ( l + k - 1 )
    # pq-tuples
    pq_t = [(2,1)] + [(2,2)] + [(3,3)] * ( l + k - 1 )

    # aA-bB Tuples
    eps_45_t = [ tuple( [1] + [fp.alpha / 2.0] * (l+k) ), tuple([ 0.5 ] + [ fp.alpha / 2 ] * ( l + k ) ) ]
    eps_6_t = [ tuple( [0] + [fp.alpha / 2.0] * (l+k) ) ]
    a_t = [[(1-fp.ms, 1), ((fp.z**2)/fp.alpha + 1,1)]] + [[(1-fp.ms, 1), (1,1), ((fp.z**2)/fp.alpha + 1,1)]] * (l+k-1)
    b_t = [[(fp.mu, 1), ((fp.z**2)/fp.alpha,1)]] + [[(fp.mu, 1), ((fp.z**2)/fp.alpha,1), (0,1)]] * (l+k-1)

    out_t = {
        'mn': mn_t,
        'pq': pq_t,
        'a': a_t,
        'b': b_t,
        'eps_45': eps_45_t,
        'eps_6': eps_6_t
    }

    return out_t

def compute_lth_user_pep( l:int, sp: NomaSystem, fp: AFFading, gamma_snr: float ):

    # Compute theta_l
    noise_power = 1
    theta_l = sp.compute_theta_lth( noise_power, l-1 )
    # Constant factor
    c1 = ( gamma( sp.num_users + 1 ) * fp.alpha ) / ( 4 * np.sqrt( np.pi ) * gamma( l ) * gamma( sp.num_users - l + 1 ) )

    # Sum terms
    h_f = 0
    for k in range(0, sp.num_users - l + 1):

        # Constant factor
        c2 = binom( sp.num_users - l, k ) * ( ( -1 )**( k ) ) * ( ( ( fp.z**2 ) / ( fp.alpha * gamma( fp.mu ) * gamma( fp.ms ) ) )**( l + k ) )
        # H-Fox parameters
        hfox_dict = get_lth_pep_hfox_tuples( l, k, fp )
        omega = [(fp.psi * ( np.sqrt(2 * fp.lambda_p) * fp.z )**(fp.alpha)) / (( ( theta_l**2 ) * gamma_snr * ( fp.z**2 + 2 ) )**(fp.alpha/2))] * ( l + k )
        hfox_params = omega, \
            hfox_dict['mn'], \
            hfox_dict['pq'], \
            hfox_dict['eps_45'], \
            hfox_dict['eps_6'], \
            hfox_dict['a'], \
            hfox_dict['b']
        # HFox computation
        h_f = h_f + c2 * np.real(compMultiFoxH(hfox_params, nsubdivisions=50, boundaryTol=1e-6))

    pep = c1 * h_f

    return pep

def compute_lth_user_pep_th_based( l:int, fp: AFFading, num_users: int, theta_l: float, gamma_snr: float ):

    # Constant factor
    c1 = ( gamma( num_users + 1 ) * fp.alpha ) / ( 4 * np.sqrt( np.pi ) * gamma( l ) * gamma( num_users - l + 1 ) )
    
    # Sum terms
    h_f = 0
    for k in range(0, num_users - l + 1):

        # Constant factor
        c2 = binom( num_users - l, k ) * ( ( -1 )**( k ) ) * ( ( ( fp.z**2 ) / ( fp.alpha * gamma( fp.mu ) * gamma( fp.ms ) ) )**( l + k ) )
        # H-Fox parameters
        hfox_dict = get_lth_pep_hfox_tuples( l, k, fp )
        omega = [(fp.psi * ( np.sqrt(2 * fp.lambda_p) * fp.z )**(fp.alpha)) / (( ( theta_l**2 ) * gamma_snr * ( fp.z**2 + 2 ) )**(fp.alpha/2))] * ( l + k )
        hfox_params = omega, \
            hfox_dict['mn'], \
            hfox_dict['pq'], \
            hfox_dict['eps_45'], \
            hfox_dict['eps_6'], \
            hfox_dict['a'], \
            hfox_dict['b']
        # HFox computation
        h_f = h_f + c2 * np.real(compMultiFoxH(hfox_params, nsubdivisions=100, boundaryTol=1e-8))

    pep = c1 * h_f

    return pep


def compute_lth_user_assym_pep( l:int, sp: NomaSystem, fp: AFFading, gamma_snr: float ):

    # Compute theta_l
    theta_l = sp.compute_theta_l(1)
    pep_asy = 0

    m, mu, z, alpha, lambda_p, psi = fp.ms, fp.mu, fp.z, fp.alpha, fp.lambda_p, fp.psi

    if fp.alpha * fp.mu <= fp.z**2:

        c1 = (math.factorial( sp.num_users ) * gamma( 0.5 + ( alpha * mu * l / 2.0 ) )) / ( 2 * l * np.sqrt( np.pi ) * math.factorial( l - 1 ) * math.factorial( sp.num_users - l ) )
        c2 = (( psi**( mu ) ) * ( z**2 ) * gamma( m + mu )) / ( (( ( z**2 ) / alpha ) - mu ) * alpha * mu * gamma( mu ) * gamma( m ) )
        c3 = (z * np.sqrt( 2.0 * lambda_p )) / ( np.sqrt( theta_l[l-1]**2 * gamma_snr * ( z**2 + 2 ) ) )
        pep_asy = c1 * ( c2 * ( c3**( alpha * mu ) ) )**( l )

    else:

        c1 = (math.factorial( sp.num_users ) * gamma( 0.5 + ( (z**2) * l / 2.0 ) )) / ( 2 * l * np.sqrt( np.pi ) * math.factorial( l - 1 ) * math.factorial( sp.num_users - l ) )
        c2 = (( psi**( (z**2)/alpha ) ) * gamma( mu - ((z**2)/alpha) ) * gamma( m + (z**2)/alpha )) / ( gamma( mu ) * gamma( m ) )
        c3 = (z * np.sqrt( 2.0 * lambda_p )) / ( np.sqrt( theta_l[l-1]**2 * gamma_snr * ( z**2 + 2 ) ) )
        pep_asy = c1 * ( c2 * ( c3**( z**2 ) ) )**( l )

    return pep_asy


def get_random_generator( fading_info: AFFading, domain: list ):

    # Random state
    urng = np.random.default_rng()
    # Generator
    rv_generator = NumericalInversePolynomial( fading_info, random_state=urng, domain=domain )

    return rv_generator

def qfunc(x):
    return 0.5-0.5*erf(x/np.sqrt(2))

def pep_monte_carlo_sim( sp: NomaSystem, noise_power: float, rv_gen, num_samples: int ):

    # Create the theta_l vector
    theta_l = sp.compute_theta_l(noise_power)

    # Generate the RV matrix
    af_rvs = rv_gen.rvs( [sp.num_users, num_samples] )
    af_rvs = np.sort(af_rvs,axis=0)

    # Create the pep vector
    mc_pep = np.ones(3)
    for l in range(0, sp.num_users):

        lth_af_fading = np.reshape(af_rvs[l, :],[num_samples,1])
        mc_pep[l] = np.mean( qfunc( np.sqrt( lth_af_fading ) * theta_l[l] ) )
    
    return mc_pep


def calc_bit_error( x: int, x_p: int ):

    def dec_to_bin(y):
        return int(bin(y)[2:])

    xor = dec_to_bin(x^x_p)
    diff = [int(x) for x in str(xor)]
    diff_length = sum(diff)
    return diff_length

def compute_union_bound( sp: NomaSystem, fp: AFFading, gamma_snr, lp:int ):

    # Create all combination between symbols
    ld_t_symbols = np.zeros( [sp.num_users, sp.qam_order * ( sp.qam_order - 1 )] ).astype(int)
    ld_r_symbols = np.zeros( [sp.num_users, sp.qam_order * ( sp.qam_order - 1 )] ).astype(int)
    t_symbols_comb = []
    r_symbols_comb = []
    for l in range(1, sp.num_users+1):
        c=0
        for m_ui in range(0, sp.qam_order):
            for m_uip in range(0, sp.qam_order):
            
                if m_ui != m_uip:
            
                    ld_t_symbols[ l-1, c ] = m_ui
                    ld_r_symbols[ l-1, c ] = m_uip
                    c = c + 1
        t_symbols_comb.append( ld_t_symbols[l-1,:] )
        r_symbols_comb.append( ld_r_symbols[l-1,:] )

    t_sym_l = list( product(*t_symbols_comb) )
    r_sym_l = list( product(*r_symbols_comb) )

    prob_per_symbol = 1.0 / sp.qam_order
    union_bound_ber = 0    
    for i in range(0,len(t_sym_l)):

        num_error_bits = calc_bit_error( t_sym_l[i][lp-1], r_sym_l[i][lp-1] )
        t_symbols = list(t_sym_l[i])
        r_symbols = list(r_sym_l[i])
        # Auxiliary system parameters
        aux_sp = NomaSystem( sp.num_users, sp.tx_power, sp.power_alloc, t_symbols, r_symbols, sp.qam_order )

        # pprint(vars(aux_sp))
        # pprint(vars(fp))
        pep_xx = compute_lth_user_pep( lp, aux_sp, fp, gamma_snr )

        union_bound_ber = union_bound_ber + ( prob_per_symbol * num_error_bits * pep_xx )

    avg_f = ( sp.qam_order * ( sp.qam_order - 1 ) )**( sp.num_users - 1 )
    union_bound_ber = union_bound_ber / ( avg_f * sp.num_bits_per_symbol )


    return union_bound_ber

def compute_union_bound2( sp: NomaSystem, fp: AFFading, gamma_snr, lp:int, c_seq ):

    prob_per_symbol = 1.0 / sp.qam_order
    union_bound_ber = 0

    peps = np.zeros( sp.qam_order )
    c = np.zeros( sp.qam_order )
    info_d = {}
    info_d_c = {}
    for i in range(0, len(c_seq)):

        t_symbols = c_seq[i][0]
        r_symbols = c_seq[i][1]

        if t_symbols[ lp - 1 ] != r_symbols[ lp - 1 ]:

            q_xx = calc_bit_error(  t_symbols[ lp - 1 ], r_symbols[ lp - 1 ] )
            aux_sp = NomaSystem( sp.num_users, sp.tx_power, sp.power_alloc, t_symbols, r_symbols, sp.qam_order )

            info_t = tuple((t_symbols) + (r_symbols))
            if info_t not in info_d:
                info_d[ info_t ] = 0
                info_d_c[ info_t ] = 0

            peps[ t_symbols[ lp - 1 ] ] = peps[ t_symbols[ lp - 1 ] ] + compute_lth_user_pep( lp, aux_sp, fp, gamma_snr ) * q_xx
            info_d[ info_t ] = info_d[ info_t ] + peps[ t_symbols[ lp - 1 ] ]
            info_d_c[ info_t ] = info_d_c[ info_t ] + 1

    # for info in info_d.keys():

    #     info_d[ info ] = info_d[ info ] / ( info_d_c[info] )

    union_bound_ber = union_bound_ber + ( prob_per_symbol ) * np.mean( list( info_d.values() ) )

    return union_bound_ber

def compute_union_bound3( sp: NomaSystem, fp: AFFading, gamma_snr, lp:int, c_seq, info_seq ):

    prob_per_symbol = 1.0 / sp.qam_order

    t_symbols = np.zeros( [len(c_seq), sp.num_users] ).astype(int)
    r_symbols = np.zeros( [len(c_seq), sp.num_users] ).astype(int)
    deltas = np.zeros( [len(c_seq), sp.num_users], dtype=np.complex128 )

    for i in range(0, len(c_seq)):
        
        t_symbols[i] = c_seq[i][0]
        r_symbols[i] = c_seq[i][1]
        deltas[i] = c_seq[i][2]

    peps_dict = {}
    peps = np.zeros( sp.qam_order )
    peps_array = np.zeros( len( c_seq ) )
    for i in range(0, len(c_seq)):

        if np.abs( deltas[ i, lp - 1 ] ) != 0:

            t_symbols_f = t_symbols[i,:]
            r_symbols_f = r_symbols[i,:]
            deltas_f = deltas[i,:]

            f_id = tuple( tuple(np.delete(t_symbols_f, lp - 1)) + tuple(np.delete(deltas_f, lp - 1)) )
            if f_id not in peps_dict:
                peps_dict[f_id] = []
            g_id = tuple( tuple(t_symbols_f) + tuple(r_symbols_f) )

            q_xx = calc_bit_error( t_symbols[i, lp - 1], r_symbols[i, lp - 1] )
            aux_sp = NomaSystem( sp.num_users, sp.tx_power, sp.power_alloc, t_symbols_f, r_symbols_f, sp.qam_order )
            pep_xx = q_xx * compute_lth_user_pep( lp, aux_sp, fp, gamma_snr ) * info_seq[ g_id ]

            peps_array[ i ] = pep_xx
            peps_dict[f_id].append( pep_xx )

    peps_c = np.zeros( len( peps_dict.keys() ) )
    c = 0
    for key, value in peps_dict.items():

        peps_c[c] = np.sum( value )
        c = c + 1

    union_bound = (np.sum( peps_c ) * (2**(lp - 1)))

    return union_bound, peps_array, peps_dict

def compute_union_bound4( sp: NomaSystem, fp: AFFading, gamma_snr, lp:int, c_seq, info_seq ):

    prob_per_symbol = 1.0 / sp.qam_order

    t_symbols = np.zeros( [len(c_seq), sp.num_users] ).astype(int)
    r_symbols = np.zeros( [len(c_seq), sp.num_users] ).astype(int)
    deltas = np.zeros( [len(c_seq), sp.num_users], dtype=np.complex128 )

    for i in range(0, len(c_seq)):
        
        t_symbols[i] = c_seq[i][0]
        r_symbols[i] = c_seq[i][1]
        deltas[i] = c_seq[i][2]

    peps_dict = {}
    peps = np.zeros( sp.qam_order )
    peps_array = np.zeros( len( c_seq ) )
    
    union_bound = 0

    for i in range(0,len(t_symbols)):

        t_id = tuple(t_symbols[i, :])
        d_id = tuple(deltas[i, 0:lp-1])
        f_id = tuple( t_id + d_id )

        if np.abs( deltas[i, lp-1] ) > 0:
            if f_id not in peps_dict:
                peps_dict[ f_id ] = []

            q_xx = calc_bit_error( t_symbols[i, lp - 1], r_symbols[i, lp - 1] )
            aux_sp = NomaSystem( sp.num_users, sp.tx_power, sp.power_alloc, t_symbols[i, :], r_symbols[i, :], sp.qam_order )
            pep_c = q_xx * compute_lth_user_pep( lp, aux_sp, fp, gamma_snr )
            peps_dict[ f_id ].append( pep_c )

    for v in peps_dict.values():

        union_bound = union_bound + np.mean( v )

    union_bound = union_bound * prob_per_symbol * ( 1 / sp.num_bits_per_symbol )

    return union_bound, peps_array, peps_dict

def ber_monte_carlo_sim_sys( sp: NomaSystem, noise_power: float, rv_gen, num_samples: int, perfect_sic=False ):

    # Generate the fading matrix
    af_rvs = rv_gen.rvs( [sp.num_users, num_samples] )
    p = np.mean( af_rvs )
    af_rvs = np.sort(af_rvs,axis=0)
    H = np.sqrt( af_rvs )
    
    # Generate Noise matrix
    n_mat = ( np.sqrt( noise_power / 2 ) ) * ( np.random.normal(0, 1, [sp.num_users, num_samples]) + 1j * np.random.normal(0, 1, [sp.num_users, num_samples]) )
    n_v = np.mean( np.abs( n_mat )**2 )

    sample_snr_db = 10 * np.log10( p / n_v )
    # Transmitted symbols list
    t_symbols_id = np.random.randint( sp.qam_order, size=[sp.num_users, num_samples] )
    # Bit error
    total_bit_error = np.zeros( sp.num_users, dtype=int )
    ber = np.zeros( sp.num_users )
    c_seq_l = [[],[],[]]
    info_cseq = {}
    pep_matrix = np.zeros( [ sp.num_users, sp.qam_order, sp.qam_order ] ).astype(int)
    for n in range(0, num_samples):

        # Channels
        h = H[:, n]
        # Noise
        w = n_mat[:, n]
        # Transmitted symbols
        t_symbols_id_n = t_symbols_id[:, n]
        t_symbols = sp.constellation[ t_symbols_id_n ]
        # Transmitted signal
        s = np.sum( np.sqrt( sp.power_alloc * sp.tx_power ) * t_symbols )

        for l in range(0, sp.num_users):

            if l not in info_cseq:
                info_cseq[l] = {}

            power_f = np.sqrt( sp.power_alloc[ l ] * sp.tx_power ) * h[l]

            r_symbols_id = [0] * sp.num_users
            # SIC
            sic=0
            if perfect_sic is False:
                for j in range(0, l):

                    rec_lp = ( s - sic ) * h[l] + w[l]
                    # Recover symbol
                    rec_sym_id = np.argmin( np.abs( (rec_lp) - (sp.constellation * power_f) )**2 )
                    rec_sym = sp.constellation[ rec_sym_id ]
                    # Add to SIC
                    sic = sic + np.sqrt( sp.power_alloc[ j ] * sp.tx_power ) * rec_sym
                    r_symbols_id[j] = rec_sym_id
            else:
                for j in range(0, l):

                    rec_lp = ( s - sic ) * h[l] + w[l]
                    # Recover symbol
                    rec_sym_id = t_symbols_id_n[j]
                    rec_sym = sp.constellation[ rec_sym_id ]
                    # Add to SIC
                    sic = sic + np.sqrt( sp.power_alloc[ j ] * sp.tx_power ) * rec_sym
                    r_symbols_id[j] = rec_sym_id

            # Rec signal
            rec_l = ( s - sic ) * h[l] + w[l]

            # Recover symbol
            rec_sym_id = np.argmin( np.abs( (rec_l) - (sp.constellation * power_f) )**2 )
            rec_sym = sp.constellation[ rec_sym_id ]
            r_symbols_id[l] = rec_sym_id

            # Number of errors
            if rec_sym_id != t_symbols_id_n[l]:
                bit_error = calc_bit_error( rec_sym_id, t_symbols_id_n[l] )
                total_bit_error[l] = total_bit_error[l] + bit_error

                f_id = tuple( tuple(t_symbols_id_n) + tuple(r_symbols_id) )
                if f_id not in info_cseq[l]:
                    info_cseq[l][f_id] = 0
                info_cseq[l][f_id] = info_cseq[l][f_id] + 1

            delta = t_symbols - sp.constellation[ r_symbols_id ]
            c_seq = (list(t_symbols_id_n),r_symbols_id,list(delta))
            if c_seq not in c_seq_l[l]:
                c_seq_l[l].append(c_seq)


            # f_id = tuple( tuple(t_symbols_id_n) + tuple(r_symbols_id) )
            # if f_id not in info_cseq[l]:
            #     info_cseq[l][f_id] = 0
            # info_cseq[l][f_id] = info_cseq[l][f_id] + (1/ num_samples)

    for l in range(0, sp.num_users):
        total = sum(list(info_cseq[l].values()))
        for key in info_cseq[l].keys():
            info_cseq[l][key] = info_cseq[l][key] / total

    ber = total_bit_error / ( num_samples * sp.num_bits_per_symbol )

    return ber, c_seq_l, sample_snr_db, info_cseq


def ber_monte_carlo_sim_sys_lth( sp: NomaSystem, noise_power: float, rv_gen, num_samples: int, lp: int ):

    # Generate the fading matrix
    af_rvs = rv_gen.rvs( [sp.num_users, num_samples] )
    af_rvs = np.sort(af_rvs,axis=0)
    H = np.sqrt( af_rvs )
    
    # Generate Noise matrix
    n_mat = ( np.sqrt( noise_power / 2 ) ) * ( np.random.normal(0, 1, [num_samples,1]) + 1j * np.random.normal(0, 1, [num_samples,1]) )
    n_v = np.mean( np.abs( n_mat )**2 )

    # Transmitted symbols list
    t_symbols_id = np.random.randint( sp.qam_order, size=[sp.num_users, num_samples] )
    r_symbols_id = np.random.randint( sp.qam_order, size=[lp - 1, num_samples] )
    # Bit error
    total_bit_error = 0
    ber = 0
    for n in range(0, num_samples):

        
        # Channels
        h = H[lp-1,n]
        print(H[:,n], h)
        # Noise
        w = n_mat[n]
        # Transmitted symbols
        t_symbols_id_n = t_symbols_id[:, n]
        t_symbols = sp.constellation[ t_symbols_id_n ]
        # Transmitted signal
        s = np.sum( np.sqrt( sp.power_alloc * sp.tx_power ) * t_symbols )
        # Power factor
        power_f = np.sqrt( sp.power_alloc[ lp - 1 ] * sp.tx_power ) * h

        # SIC
        sic = 0
        for j in range(0, lp - 1):
            # Recovered symbol
            rec_sym_id = r_symbols_id[ j, n ]
            rec_sym = sp.constellation[ rec_sym_id ]
            # Add to SIC
            sic = sic + np.sqrt( sp.power_alloc[ j ] * sp.tx_power ) * rec_sym

        # Rec signal
        rec_l = ( s - sic ) * h + w
        # Recover symbol
        rec_sym_id = np.argmin( np.abs( (rec_l) - (sp.constellation * power_f) )**2 )
        rec_sym = sp.constellation[ rec_sym_id ]
        # Number of errors
        bit_error = calc_bit_error( rec_sym_id, t_symbols_id_n[lp-1] )
        total_bit_error = total_bit_error + bit_error

    ber = total_bit_error / ( num_samples * sp.num_bits_per_symbol )

    return ber