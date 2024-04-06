import argparse
import exp

parse = argparse.ArgumentParser(description= "Using attention mechanic for classification")
parse.add_argument("--input_shape", type= list, help = "Shape of input", default= [11,1])
parse.add_argument("--d_model", type = int, help = "dimenson of embedded input", default= 11)
parse.add_argument("--d_atten", type= int, help = "dimenson of input attention", default= 13)
parse.add_argument("--num_of_head", type = int, help = "number head attention", default= 5)
parse.add_argument("--num_of_l2", type= int, help = "quantity of unit in fully connected", default= 11)
parse.add_argument("--num_of_multi", type= int, help = "num of layer multiatten", default= 5)
parse.add_argument("--output_shape", type= int, help = "number of label", default= 11)
parse.add_argument("--type_active", type= str, help = "active function to output", default= "softmax")
parse.add_argument("--filepath", type = str, help = "path to data", default = "data_clean.csv")
parse.add_argument("--target", type = int, help = "index column to predict", default = 11)
parse.add_argument("--batch_size", help = "size of batch data", default = 64, type = int )
parse.add_argument("--epoch", type = int, help = "epoch", default = 5)
args = parse.parse_args()

if __name__ == "__main__":
    exp_run = exp.Exp_Wine(args)
    exp_run.getdata()
    exp_run.train_test_loop()