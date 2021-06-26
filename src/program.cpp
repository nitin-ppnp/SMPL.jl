// This file is a part of Julia. License is MIT: http://julialang.org/license

// Standard headers
#include <string.h>
#include <stdint.h>
// #include <errno.h>
#include <iostream>

#ifdef JULIA_DEFINE_FAST_TLS // only available in Julia v0.7 and above
JULIA_DEFINE_FAST_TLS()
#endif

#define JULIA_ENABLE_THREADING 0
// Julia headers (for initialization and gc commands)
#include "uv.h"
#include "julia.h"

#include <chrono>

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

struct SMPL {
	jl_array_t* v_template;
	jl_array_t* shapedirs;
	jl_array_t* posedirs;
	jl_array_t* J_regressor;
	jl_array_t* parents;
	jl_array_t* lbs_weights;
	jl_array_t* f;
};

extern "C" {
	// Declare C prototype of a function defined in Julia
	extern jl_array_t* CSMPL_v_template(jl_array_t*);
    extern jl_array_t* CSMPL_shapedirs(jl_array_t*);
    extern jl_array_t* CSMPL_posedirs(jl_array_t*);
    extern jl_array_t* CSMPL_J_regressor(jl_array_t*);
    extern jl_array_t* CSMPL_parents(jl_array_t*);
    extern jl_array_t* CSMPL_lbs_weights(jl_array_t*);
    extern jl_array_t* CSMPL_f(jl_array_t*);
    //extern jl_array_t* test(jl_value_t*);
	extern jl_array_t* CSMPL_get_normals(jl_value_t*,
										jl_value_t*,
										jl_value_t*);
    extern jl_value_t* CSMPL_LBS(jl_value_t*,
								jl_value_t*,
								jl_value_t*,
								jl_value_t*,
								jl_value_t*,
								jl_value_t*,
								jl_value_t*,
								jl_value_t*,
								jl_value_t*,
								jl_value_t*);
	// extern void jl_init_with_image(void *, const char *);
	// extern void jl_atexit_hook(int);
}


// main function (windows UTF16 -> UTF8 argument conversion code copied from julia's ui/repl.c)
int main(int argc, char *argv[])
{
	int i;
	uv_setup_args(argc, argv); // no-op on Windows

	// // initialization
	// libsupport_init();

	// // jl_options.compile_enabled = JL_OPTIONS_COMPILE_OFF;
	// // JULIAC_PROGRAM_LIBNAME defined on command-line for compilation
	// jl_options.image_file = JULIAC_PROGRAM_LIBNAME;
	// julia_init(JL_IMAGE_JULIA_HOME);
    // "D:\\julia\\projects\\smpl\\builddir\\SMPLMod.dll"
	char* csmpl_lib_path(getenv("CSMPL_LIB_PATH"));

	jl_init_with_image__threading(NULL,csmpl_lib_path);
	// // Initialize Core.ARGS with the full argv.
	jl_set_ARGS(argc, argv);

	// // Set PROGRAM_FILE to argv[0].
	jl_set_global(jl_base_module,
		jl_symbol("PROGRAM_FILE"), (jl_value_t*)jl_cstr_to_string(argv[0]));

	// // Set Base.ARGS to `String[ unsafe_string(argv[i]) for i = 1:argc ]`
	jl_array_t *ARGS = (jl_array_t*)jl_get_global(jl_base_module, jl_symbol("ARGS"));
	jl_array_grow_end(ARGS, argc - 1);
	for (i = 1; i < argc; i++) {
		jl_value_t *s = (jl_value_t*)jl_cstr_to_string(argv[i]);
		std::cout << argv[i] << std::endl;
		jl_arrayset(ARGS, s, i - 1);
	}
    // timestamp_t t0 = get_timestamp();

    // jl_value_t* smpl_path = jl_cstr_to_string("D:\\julia\\projects\\smpl\\smpl_new.npz");	// call the work function, and get back a value
	//JL_GC_PUSH1(smpl_path);

	float* v_template_buf = new float[6890*3];
	float* shapedirs_buf = new float[6890*3*10];
	float* posedirs_buf = new float[207*20670];
	float* J_regressor_buf = new float[24*6890];
	unsigned int* parents_buf = new unsigned int[24];
	float* lbs_weights_buf = new float[6890*24];
	unsigned int* f_buf = new unsigned int[13776*3];

	float* pose_buf = new float[72];
	float* shape_buf = new float[10];
	float* trans_buf = new float[3];
	float* verts_buf = new float[6890*3];
	float* verts_normals_buf = new float[6890*3];
	std::fill_n(pose_buf,72,0);
	std::fill_n(shape_buf,10,0);
	std::fill_n(trans_buf,3,0);
	std::fill_n(verts_buf,6890*3,0);
	std::fill_n(verts_normals_buf,6890*3,0);


	jl_value_t* jl_fl32_1_arr = jl_apply_array_type((jl_value_t*)jl_float32_type, 1);
	jl_value_t* jl_fl32_2_arr = jl_apply_array_type((jl_value_t*)jl_float32_type, 2);
	jl_value_t* jl_fl32_3_arr = jl_apply_array_type((jl_value_t*)jl_float32_type, 3);
	jl_value_t* jl_uint32_2_arr = jl_apply_array_type((jl_value_t*)jl_uint32_type, 2);
	jl_value_t* jl_uint32_1_arr = jl_apply_array_type((jl_value_t*)jl_uint32_type, 1);

	jl_array_t* v_template = jl_ptr_to_array(jl_fl32_2_arr, v_template_buf, (jl_value_t*)jl_eval_string("(6890,3)"), 0);
	jl_array_t* shapedirs = jl_ptr_to_array(jl_fl32_3_arr, shapedirs_buf, (jl_value_t*)jl_eval_string("(6890,3,10)"), 0);
	jl_array_t* posedirs = jl_ptr_to_array(jl_fl32_2_arr, posedirs_buf, (jl_value_t*)jl_eval_string("(207,20670)"), 0);
	jl_array_t* J_regressor = jl_ptr_to_array(jl_fl32_2_arr, J_regressor_buf, (jl_value_t*)jl_eval_string("(24,6890)"), 0);
	jl_array_t* parents = jl_ptr_to_array_1d(jl_uint32_1_arr, parents_buf, 24, 0);
	jl_array_t* lbs_weights = jl_ptr_to_array(jl_fl32_3_arr, lbs_weights_buf, (jl_value_t*)jl_eval_string("(6890,24)"), 0);
	jl_array_t* f = jl_ptr_to_array(jl_uint32_2_arr, f_buf, (jl_value_t*)jl_eval_string("(13776,3)"), 0);
	jl_array_t* pose = jl_ptr_to_array_1d(jl_fl32_1_arr, pose_buf, 72, 0);
	jl_array_t* shape = jl_ptr_to_array_1d(jl_fl32_1_arr, shape_buf, 10, 0);
	jl_array_t* trans = jl_ptr_to_array_1d(jl_fl32_1_arr, trans_buf, 3, 0);
	jl_array_t* smpl_verts = jl_ptr_to_array(jl_fl32_2_arr, verts_buf, (jl_value_t*)jl_eval_string("(3,6890)"), 0);
	jl_array_t* verts_normals = jl_ptr_to_array(jl_fl32_2_arr, verts_normals_buf, (jl_value_t*)jl_eval_string("(3,6890)"), 0);

	jl_gc_enable(0);
    jl_array_t* smpl_v_template = CSMPL_v_template(v_template);
	jl_array_t* smpl_shapedirs = CSMPL_shapedirs(shapedirs);
	jl_array_t* smpl_posedirs = CSMPL_posedirs(posedirs);
	jl_array_t* smpl_J_regressor = CSMPL_J_regressor(J_regressor);
	jl_array_t* smpl_lbs_weights = CSMPL_lbs_weights(lbs_weights);
	jl_array_t* smpl_parents = CSMPL_parents(parents);
	jl_array_t* smpl_f = CSMPL_f(f);

	jl_gc_enable(1);

	std::cout << "asffffffffffffsfasfffffasfafafs" << std::endl;
    
	for (int i=0;i<10;i++)
    {
        std::cout << verts_normals_buf[i] << std::endl;
    }
	
	Timer tmr;
    tmr.reset();
	jl_gc_enable(0);
    jl_array_t* verts = (jl_array_t*)CSMPL_LBS((jl_value_t*)v_template, (jl_value_t*)shapedirs, (jl_value_t*)posedirs, (jl_value_t*)J_regressor, 
	(jl_value_t*)parents, (jl_value_t*)lbs_weights,(jl_value_t*)pose,(jl_value_t*)shape,(jl_value_t*)trans,(jl_value_t*)smpl_verts);
	jl_array_t* normals = (jl_array_t*)CSMPL_get_normals((jl_value_t*)smpl_verts,(jl_value_t*)smpl_f,(jl_value_t*)verts_normals);
	jl_gc_enable(1);
	
	double t1 = tmr.elapsed();
    float* data = (float*)jl_array_data(smpl_verts);

    std::cout << std::endl << t1 << std::endl;
/*
    for (int i=0;i<10;i++)
    {
        std::cout << verts_normals_buf[i] << std::endl;
    }

    /*

    float pose[100];
    memset(pose,0,72);

    for(int p=0;p<72;p++)
    {
        pose[p] = 0;
        std::cout << pose[p] << "" ;
    }
    jl_value_t* jl_pose_type = jl_apply_array_type((jl_value_t*)jl_float32_type, 1);
	jl_array_t* jl_pose = jl_ptr_to_array_1d(jl_pose_type,pose,72,0);

    Timer tmr;
    double t1 = tmr.elapsed();
    tmr.reset();
    jl_array_t* verts = CSMPL_LBS(retcode);
    double t2 = tmr.elapsed();
    // SMPL* smplobj = (SMPL*)retcode; 
	float* js = (float*)jl_array_data(verts);
	int ndim1 = jl_array_dim(verts, 0);
	int ndim2 = jl_array_dim(verts, 1);
	std::cout << std::endl << std::endl;
	for (i = 0; i < 10; i++) {
		for (int j = 0; j < 2; j++) {
			std::cout << js[j + 2*i] << "  ";
		}
		std::cout << std::endl;
	}

    std::cout << std::endl << t1 << std::endl;
    std::cout << std::endl << t2  << std::endl;

*/
	// Cleanup and gracefully exit
	jl_atexit_hook(0);
	return 0;
}
