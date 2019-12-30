set terminal jpeg;
set output "neuralnet.jpg";
set title "Neural Network Loss By Epoch";
set xlabel "Epoch";
set ylabel "Loss";
plot "nn.txt" using ($0):2 title 'Model' with lines, \

