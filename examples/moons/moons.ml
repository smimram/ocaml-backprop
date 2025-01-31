open Backprop

let moons =
  let samples = 200 in
  let data = ref [] in
  for _ = 1 to samples do
    let a = Random.float Float.pi in
    let r = 0.25 +. Random.float 0.1 in
    let x = r *. cos a +. 0.35 in
    let y = r *. sin a +. 0.5 in
    data := ((x,y),true) :: !data
  done;
  for _ = 1 to samples do
    let a = Random.float Float.pi in
    let r = 0.25 +. Random.float 0.1 in
    let x = r *. cos a +. 0.65 in
    let y = -. (r *. sin a) +. 0.6 in
    data := ((x,y),false) :: !data
  done;
  !data

let net =
  let layer1 =
    let weights = ref @@ Vector.Linear.uniform 2 16 in
    let bias = ref @@ Vector.zero 16 in
    Backpropagatable.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  let layer2 =
    let weights = ref @@ Vector.Linear.uniform 16 16 in
    let bias = ref @@ Vector.zero 16 in
    Backpropagatable.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  let layer3 =
    let weights = ref @@ Vector.Linear.uniform 16 2 in
    let bias = ref @@ Vector.zero 2 in
    Backpropagatable.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  fun x -> x |> layer1 |> layer2 |> layer3

let () =
  Random.self_init ();
  let window = 500 in
  let plot (x,y) b =
    let x = x *. float window |> int_of_float in
    let y = y *. float window |> int_of_float in
    let c = if b then Graphics.green else Graphics.red in
    Graphics.set_color c;
    Graphics.fill_circle x y 2
  in
  Graphics.open_graph "";
  Graphics.resize_window window window;
  List.iter (fun ((x,y),b) -> plot (x,y) b) moons;
  Graphics.loop_at_exit [Button_down; Key_pressed] (fun _ -> raise Exit)
