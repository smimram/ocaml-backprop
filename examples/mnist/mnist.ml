let () =
  print_endline "Welcome to MNIST.";
  let labels = Idx.load_labels "data/train-labels-idx1-ubyte" in
  Printf.printf "loaded %d labels\n%!" (Array.length labels);
  let images = Idx.load_images "data/train-images-idx3-ubyte" in
  Printf.printf "loaded %d images\n%!" (Array.length images);
  ()
