(** The MNIST dataset. *)

(* See for instance https://nextjournal.com/gkoehler/pytorch-mnist *)

let () =
  let limit = 1000 in
  print_endline "Welcome to MNIST.";
  let labels = Idx.load_labels "data/train-labels-idx1-ubyte" in
  Printf.printf "loaded %d labels\n%!" (Array.length labels);
  let images = Idx.load_images ~limit "data/train-images-idx3-ubyte" in
  Printf.printf "loaded %d images\n%!" (Array.length images);

  let img = images.(0) in
  let height = Array.length img in
  let width = Array.length img.(0) in
  Printf.printf "image size: %d * %d\n%!" width height;

  let scale = 10 in
  Graphics.open_graph "";
  Graphics.resize_window (scale*width) (scale*height);
  for n = 0 to limit - 1 do
    let img = images.(n) in
    for i = 0 to width - 1 do
      for j = 0 to height - 1 do
        let c = img.(height - 1 - j).(i) in
        let c = Graphics.rgb c c c in
        Graphics.set_color c;
        Graphics.fill_rect (scale*i) (scale*j) scale scale
      done
    done;
  done;
  Graphics.loop_at_exit [Button_down; Key_pressed] (fun _ -> raise Exit)

