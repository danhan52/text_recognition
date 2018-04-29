import sys
import os

# for getting rid of previous models (to save space)
def remove_old_ckpt(b, output_model_dir):
    mdl_base = output_model_dir+"model" + b + ".ckpt"
    try:
        os.remove(mdl_base+".data-00000-of-00001")
    except:
        pass
    
    try:
        os.remove(mdl_base+".index")
    except:
        pass

    try:
        os.remove(mdl_base+".meta")
    except:
        pass

    try:
        os.remove(output_model_dir + "metrics_batch" + b + ".csv")
        os.remove(output_model_dir + "metrics_image" + b + ".csv")
    except:
        pass
    
    return



if __name__ == "__main__":
    if len(sys.argv) >= 4:
        b = str(int(sys.argv[1]) - int(sys.argv[3]))
        remove_old_ckpt(b=b, output_model_dir=sys.argv[2])
    print('Training Group {0} Finished!'.format(sys.argv[1]))
	print("********************************************************************************")
	print("********************************************************************************")
	print("********************************************************************************")
	print("********************************************************************************")
	print("********************************************************************************")
	print("********************************************************************************")
	print("********************************************************************************")
	print("********************************************************************************")
