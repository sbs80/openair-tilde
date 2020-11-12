/*
 * openair~.c
 *
 * A convolution reverb library for Pure Data (Pd) that allows the user
 * to switch impulse responses on the fly and responds by smoothly cross-fading
 * between the old and new convolution streams. It is designed to be used in
 * immersive simulation such as virtual reality or gaming.
 * 
 * The openair Pd library is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This software is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Parts of this code are based on by Ben Saylors partconv~ external,
 * the code for which can be found at
 * http://puredata.info/Members/bensaylor/partconv~-0.1.tar.gz/file_view
 */

#include <string.h>
#include <math.h>

#include <fftw3.h>

#include "m_pd.h"

//This is the maximum number of sections from the IR that we allow to store in the memory
#define N_SECTIONS 256

//This scaling factor might be needed to avoid clipping in the convolved output
#define SCALEFACTOR 2

//This is the number
#define FADE_OUT_LENGTH 2
#define FADE_IN_LENGTH 1

static t_class *openair_tilde_class;

typedef struct _openair_tilde
{
	t_object x_obj;

	//Buffer sizes
	t_int buffer_block_size;
	t_int buffer_FFT_size;
	t_float scale;

	t_symbol *table_name, *table_name_reserve;
	t_garray *tableObj;
	t_word *table;
	int table_size;
	
	//pointers for the complex data
	fftwf_complex *input_buffer_fft, *convolved;

	//for the impulse responses. There are two pointers for the impulse response switching algorithm.
	fftwf_complex *ir_section_fft[N_SECTIONS], *ir_section_fft_reserve[N_SECTIONS];

	//fftwf plans for fft and ifft.
	//FFT for both the impulse response and the dry sound and IFFT for the convolved sound.
	fftwf_plan plan_forward_impulse, plan_forward_input, plan_backward;

	//Initiate pointers for an output buffer and a temporary buffer
	t_float *output;
	t_float *convolved_ifft;

	//output buffer position counter
	t_int current_output_count;

	//Size of output buffer
	t_int output_size;

	//Initiate input buffer pointers
	t_float *input_buffer;
	t_float *ir_section[N_SECTIONS], *ir_section_reserve[N_SECTIONS];

	//counting variables for the DSP
	t_int count_impulse, count_dry, count_add;

	//integers to store the number of block sections that make up the impulse response
	t_int n_ir_sections, n_ir_sections_reserve;

	//size of padded buffer sections
	t_int padded_fft_size;

	//no. of bins in the FFT
	t_int n_bins;

	//fade in/out countdowns
	t_int fade_in_countdown;
	t_int fade_out_countdown;

	//fade in/out gains
	t_float in_gain, out_gain;

	//stores information about the current switching status
	t_int status;
	t_int switch_pending;

	t_sample f;

} t_openair_tilde;

//Free memory
static void openair_tilde_free(t_openair_tilde *x)
{
	t_int p;

	//free the output buffer
	freebytes ( x->output, x->output_size ) ;

	//destroy the fftwf plans
	fftwf_destroy_plan ( x->plan_forward_input );
	fftwf_destroy_plan( x->plan_backward );
	
	fftwf_destroy_plan(x->plan_forward_impulse);

 	//for each impulse response section
	for (p = 0; p < N_SECTIONS; p++)
	{
		//freebytes ( x->ir_section[p], sizeof ( x->ir_section[p] ) ) ;
		fftwf_free( x->ir_section_fft_reserve[p] );
	}

	//for each impulse response section
	for (p = 0; p < N_SECTIONS; p++)
	{
		//freebytes ( x->ir_section[p], sizeof ( x->ir_section[p] ) ) ;
		fftwf_free( x->ir_section_fft[p] );
	}
	fftwf_free( x->input_buffer_fft );
	fftwf_free( x->convolved );
}

//get and prepare the impulse response array
static void openair_tilde_set(t_openair_tilde *x, t_symbol *s)
{
	t_int table_pos, i, j;

	//get the table array from pd
	x->table_name = s;
	if ( ! (x->tableObj = (t_garray *)pd_findbyclass(x->table_name, garray_class)))
	{
 		if (*x->table_name->s_name) pd_error(x, "openair~: %s: table not found", x->table_name->s_name);
		else pd_error(x, "openair~: table not found");
		x->table = NULL;
		x->table_size = 0;
	}
	else if ( ! garray_getfloatwords(x->tableObj, &x->table_size, &x->table))
	{
		pd_error(x, "openair~: %s: bad template", x->table_name->s_name);
		x->table = NULL;
		x->table_size = 0;
	}

	//caculate the number of sections required to partition the impulse response
	x->n_ir_sections = x->table_size / x->buffer_block_size;

	if (x->table_size % x->buffer_block_size != 0)
		x->n_ir_sections++;
	if (x->n_ir_sections > N_SECTIONS)
		x->n_ir_sections = N_SECTIONS;

	//prepare each impulse response section
	for (table_pos = 0, i = 0; i < x->n_ir_sections; i++)
	{
		//set all to zeros (for padding)
		memset (x->ir_section[i], 0, sizeof(t_float) * x->padded_fft_size ) ;
	
		//fill the first part of the array with the IR section
		for ( j = 0; j < x->buffer_block_size && table_pos < x->table_size; j++, table_pos++)
		{
			x->ir_section[i][j] = x->table[table_pos].w_float;
		}

		//execute the plan to convert impulse to frequency domain with new arrays
		fftwf_execute_dft_r2c(x->plan_forward_impulse, x->ir_section[i], x->ir_section_fft[i]);
	}
	
	x->current_output_count = 0;
	
	//set to zero
	memset (x->output, 0, sizeof(t_float) * x->output_size );

	post("openair~: using %s in %d partitions with FFT-size %d", x->table_name->s_name, x->n_ir_sections, x->buffer_FFT_size);
}

//switch to new impulse response array
static void openair_tilde_startswitch(t_openair_tilde *x, t_symbol *s)
{
	t_int table_pos, i, j;

	//first of all need to copy the old impulse response data to the reserve memory buffers
	//for each impulse response section
	for (i = 0; i < x->n_ir_sections; i++)
	{

		memcpy(x->ir_section_reserve[i], x->ir_section[i], (sizeof(t_float) * x->padded_fft_size));
	}

	x->n_ir_sections_reserve = x->n_ir_sections;

	//then we need to load up the new impulse response from the buffer
	//get the table array from pd

	x->table_name = s;
	if ( ! (x->tableObj = (t_garray *)pd_findbyclass(x->table_name, garray_class)))
	{
 		if (*x->table_name->s_name) pd_error(x, "openair~: %s: table not found", x->table_name->s_name);
		else pd_error(x, "openair~: table not found");
		x->table = NULL;
		x->table_size = 0;
	}
	else if ( ! garray_getfloatwords(x->tableObj, &x->table_size, &x->table))
	{
		pd_error(x, "openair~: %s: bad template", x->table_name->s_name);
		x->table = NULL;
		x->table_size = 0;
	}
	
	//caculate the number of sections required to partition the impulse response
	x->n_ir_sections = x->table_size / x->buffer_block_size;

	if (x->table_size % x->buffer_block_size != 0)
		x->n_ir_sections++;
	if (x->n_ir_sections > N_SECTIONS)
		x->n_ir_sections = N_SECTIONS;

	//prepare each impulse response section
	for (table_pos = 0, i = 0; i < x->n_ir_sections; i++)
	{
		memset (x->ir_section[i], 0, sizeof(t_float) * x->padded_fft_size ) ;
	
		for ( j = 0; j < x->buffer_block_size && table_pos < x->table_size; j++, table_pos++)
		{
			x->ir_section[i][j] = x->table[table_pos].w_float;
		}

		fftwf_execute_dft_r2c(x->plan_forward_impulse, x->ir_section[i], x->ir_section_fft[i]);
	}

	post("openair~: SWITCH! using %s in %d partitions with FFT-size %d", x->table_name->s_name, x->n_ir_sections, x->buffer_FFT_size);
}

//stop the switch to new impulse response array
static void openair_tilde_stop_switch(t_openair_tilde *x)
{
	post("SWITCH finished");
}

//************************************************************************************************************
//PERFORM  ***************************************************************************************************
//************************************************************************************************************
static t_int *openair_tilde_perform(t_int *w)
{
	t_openair_tilde *x = (t_openair_tilde *)(w[1]);
	t_float *in = (t_float *)(w[2]);
	t_float *out = (t_float *)(w[3]);
	t_int n = (t_int)(w[4]);

	t_int i, p, j;

	//convolve the entire input sound with each section of the impulse response, block by block
	//Read from the input sound buffer, and zero pad to FFT length
	memset (x->input_buffer, 0, sizeof (t_float) * x->padded_fft_size) ;

	//status 3 means a new impulse response has been queued for switching
	if (x->status == 3)
	{
		post("loading pending impulse response");
		openair_tilde_startswitch(x, x->table_name_reserve);

		//status 2 means the switch has started
		x->status = 2;
		x->switch_pending = 0;

		//set gain of the original impulse to 1
		x->out_gain = 1;

		//set gain of the incoming impulse to 0
		x->in_gain = 0;

		//reset countdowns
		x->fade_out_countdown = FADE_OUT_LENGTH;
		x->fade_in_countdown = FADE_IN_LENGTH;
	}
	
	//if the switch is in progress, need to convolve the reserve buffer as well
	if (x->status == 2)
	{
		//fade out the old impulse response using the reserve buffer
		t_float step = 1.0/(n*FADE_OUT_LENGTH);

		for (i=0; i < n; i++)
		{
			x->input_buffer[i] = in[i]*x->out_gain;
			x->out_gain-=step;
		}

		//Do the convolution on the reserve buffer

		//FFT the dry sound section
		fftwf_execute ( x->plan_forward_input );	

		//for each impulse response section
		for (p = 0; p < x->n_ir_sections_reserve; p++)
		{
			//x->out_gain = thisGain;
			//convolve the 2, complex multiply the whole section..
			for ( i = 0 ; i < x->n_bins ; i ++ )
			{
				//real number:				
				x->convolved[i][0] = ( x->input_buffer_fft[i][0] * x->ir_section_fft_reserve[p][i][0] - x->input_buffer_fft[i][1] * x->ir_section_fft_reserve[p][i][1] ) * x->scale ;
				//imaginary number
				x->convolved[i][1] = ( x->input_buffer_fft[i][0] * x->ir_section_fft_reserve[p][i][1] + x->input_buffer_fft[i][1] * x->ir_section_fft_reserve[p][i][0] ) * x->scale ;
			}

			//inverse transform
			fftwf_execute( x->plan_backward );
						
			//add the newly convolved data to the outputbuffer
			j = (x->current_output_count + (p * x->buffer_block_size)) % x->output_size;
			for (i = 0; i < x->padded_fft_size ; i++, j = (j + 1) % x->output_size)
			{
				x->output[j] += x->convolved_ifft[i];
			}	
		}

		x->fade_out_countdown --;

		if (x->fade_out_countdown == 0)
		{
			openair_tilde_stop_switch(x);

			//TODO.. this doesn't consider if the fadeout is shorter than the fadein
			if (x->switch_pending == 0)
			{
				x->status = 1;
			}
			else
			{
				x->status = 3;
			}
		}

		//Store the input into the input buffer, and zero pad to FFT length
		memset (x->input_buffer, 0, sizeof (t_float) * x->padded_fft_size) ;

		//and fade in the new one on the main buffer
		if (x->fade_in_countdown != 0 )
		{
			step = 1.0/(n*FADE_IN_LENGTH);
			for (i=0; i < n; i++)
			{
				x->input_buffer[i] = in[i]*x->in_gain;
				x->in_gain+=step;
			}
			x->fade_in_countdown --;
		}
		else
		{
			for ( i=0; i < n; i++)
			{
				x->input_buffer[i] = in[i];
			}
		}
	}
	//else we are not switching
	else
	{
		for ( i=0; i < n; i++)
		{
			x->input_buffer[i] = in[i];
		}
	}

	//FFT the dry sound section
	fftwf_execute ( x->plan_forward_input );
	
	//for each impulse response section
	for (p = 0; p < x->n_ir_sections; p++)
	{
		//convolve the 2, complex multiply the whole section..
		for ( i = 0 ; i < x->n_bins ; i ++ )
		{
			//real number:				
			x->convolved[i][0] = ( x->input_buffer_fft[i][0] * x->ir_section_fft[p][i][0] - x->input_buffer_fft[i][1] * x->ir_section_fft[p][i][1] ) * x->scale ;
			//imaginary number
			x->convolved[i][1] = ( x->input_buffer_fft[i][0] * x->ir_section_fft[p][i][1] + x->input_buffer_fft[i][1] * x->ir_section_fft[p][i][0] ) * x->scale ;
		}

		//inverse transform
		fftwf_execute( x->plan_backward );

		//add the newly convolved data to the output buffer
		j = (x->current_output_count + (p * x->buffer_block_size)) % x->output_size;
		for (i = 0; i < x->padded_fft_size; i++, j = (j + 1) % x->output_size)
		{
			x->output[j] += x->convolved_ifft[i];
		}
	}


	//output the current section of the output buffer
	for ( i = 0, j = x->current_output_count ; i < n; i++, j = (j + 1) % x->output_size) 
	{
		out[i] = x->output[j];
		x->output[j] = 0;
	}
	

	//increment output buffer position counter
	x->current_output_count += x->buffer_block_size;
	x->current_output_count %= x->output_size;

	return (w+5);
}

//switch to new impulse response array
static void openair_tilde_trigswitch(t_openair_tilde *x, t_symbol *s)
{
	//if already switching, need to handle this
	if (x->status > 1)
	{
		post("switch pending...");
		x->table_name_reserve = s;
		x->switch_pending = 1;
		return;
	}
	//status 2 implies the switch has started
	x->status = 2;

	//reset the gains
	x->out_gain = 1;
	x->in_gain = 0;

	//reset the countdowns
	x->fade_out_countdown = FADE_OUT_LENGTH;
	x->fade_in_countdown = FADE_IN_LENGTH;

	openair_tilde_startswitch(x, s);
	
}



static void openair_tilde_dsp(t_openair_tilde *x, t_signal **sp)
{

	//load the impulse response, if not done already
	if (x->status == 0)
	{
		openair_tilde_set(x, x->table_name);
		x->status = 1;
	}
	
	dsp_add(openair_tilde_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

static void *openair_tilde_new(t_symbol *s, int argc, t_atom *argv)
{

	t_openair_tilde *x = (t_openair_tilde *)pd_new(openair_tilde_class);

	outlet_new(&x->x_obj, gensym("signal"));

	if (argc != 2)
	{
		post("argc = %d", argc);
		error("openair~: usage: [openair~ <arrayname> <blocksize>] \n\t - blocksize must be the same as the blocksize of the patch or subpatch, if different to the main patch.");
		return NULL;
	}

	x->table_name = atom_getsymbol(argv);

	x->buffer_block_size = atom_getfloatarg(1, argc, argv);

	x->buffer_FFT_size = 2 * x->buffer_block_size;

	x->scale = 1 / (float)(SCALEFACTOR*x->buffer_FFT_size);

	//need 2*(n/2+1) float array for in-place transform, where n is fftsize.
	x->padded_fft_size = 2 * (x->buffer_FFT_size / 2 + 1);

	//number of bins in fft = half of the padded size
	x->n_bins = x->buffer_FFT_size / 2 + 1;

	//set the status to 0
	x->status = 0;

	//set switch pending to 0
	x->switch_pending = 0;

	
	t_int table_pos, i;

	//allocate memory for the padded impulse response section
	for (table_pos = 0, i = 0; i < N_SECTIONS; i++)
	{
		x->ir_section[i] = fftwf_malloc(sizeof(t_float) * x->padded_fft_size) ;
		
		//complex pointer to same array
		x->ir_section_fft[i] = (fftwf_complex *) x->ir_section[i];

		//allocate memory for reserve ir
		x->ir_section_reserve[i] =  fftwf_malloc(sizeof(t_float) * x->padded_fft_size);
		x->ir_section_fft_reserve[i] = (fftwf_complex *) x->ir_section_reserve[i];
	}

	//allocate memory for the padded input buffer sections
	x->input_buffer = fftwf_malloc(sizeof(t_float) * x->padded_fft_size);	

	//complex pointer to the same array
	x->input_buffer_fft = (fftwf_complex *) x->input_buffer;			

	//Make FFT forward plan
	x->plan_forward_input =  fftwf_plan_dft_r2c_1d(x->buffer_FFT_size, x->input_buffer, x->input_buffer_fft, FFTW_MEASURE);

	//Make FFT forward plans
	x->plan_forward_impulse  = fftwf_plan_dft_r2c_1d( x->buffer_FFT_size, x->ir_section[0], x->ir_section_fft[0] , FFTW_ESTIMATE );

	//allocate a buffer to store the convolved sound and IFFT
	x->convolved_ifft = fftwf_malloc(sizeof(t_float) * x->padded_fft_size);
	x->convolved = (fftwf_complex *) x->convolved_ifft;
	x->plan_backward = fftwf_plan_dft_c2r_1d(x->buffer_FFT_size, x->convolved, x->convolved_ifft, FFTW_MEASURE);

	//set up circular output buffer
	x->output_size = x->buffer_block_size * (N_SECTIONS + 1);
	x->output = getbytes(sizeof(t_float) * x->output_size);
	
	return (x);
}

void openair_tilde_setup(void)
{
	openair_tilde_class = class_new(gensym("openair~"), (t_newmethod)openair_tilde_new, (t_method)openair_tilde_free, sizeof(t_openair_tilde), 0, A_GIMME, 0);
	
	class_addmethod(openair_tilde_class, (t_method) openair_tilde_dsp, gensym("dsp"), 0);
	class_addmethod(openair_tilde_class, (t_method) openair_tilde_trigswitch, gensym("switch"), A_DEFSYMBOL, 0);

	CLASS_MAINSIGNALIN(openair_tilde_class, t_openair_tilde, f);
}

