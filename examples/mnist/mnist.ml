let () =
  print_endline "Welcome to MNIST.";
  let labels = Idx.load_labels "data/train-labels-idx1-ubyte" in
  Printf.printf "loaded %d labels\n%!" (Array.length labels);
  let images = Idx.load_images ~limit:20 "data/train-images-idx3-ubyte" in
  Printf.printf "loaded %d images\n%!" (Array.length images);
  Graphics.open_graph "";
  let img = images.(2) in
  let height = Array.length img in
  let width = Array.length img.(0) in
  Printf.printf "image size: %d * %d\n%!" width height;
  Graphics.resize_window width height;
  for i = 0 to width - 1 do
    for j = 0 to height - 1 do
      let c = img.(height - 1 - j).(i) in
      let c = Graphics.rgb c c c in
      Graphics.set_color c;
      Graphics.plot i j 
    done
  done;
  Graphics.loop_at_exit [Button_down; Key_pressed] (fun _ -> raise Exit)

