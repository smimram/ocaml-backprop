open Backprop

let () = Printexc.record_backtrace true

let moons =
  let samples = 200 in
  let data = ref [] in
  for _ = 1 to samples do
    let a = Random.float Float.pi in
    let r = 0.25 +. Random.float 0.1 in
    let x = r *. cos a +. 0.35 in
    let y = r *. sin a +. 0.5 in
    data := ((x,y),1.) :: !data
  done;
  for _ = 1 to samples do
    let a = Random.float Float.pi in
    let r = 0.25 +. Random.float 0.1 in
    let x = r *. cos a +. 0.65 in
    let y = -. (r *. sin a) +. 0.6 in
    data := ((x,y),0.) :: !data
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

let train () =
  List.iter
    (fun (p,b) ->
       let out =
         Vector.pair p
         |> Backpropagatable.cst
         |> net
         |> Backpropagatable.Vector.squared_distance_to (Vector.scalar b)
         |> Backpropagatable.descent 0.1
       in
       Backpropagatable.run out
    ) moons

let () =
  Random.self_init ();
  let window = 500 in
  let plot (x,y) b =
    let x = x *. float window |> int_of_float in
    let y = y *. float window |> int_of_float in
    let c = Graphics.rgb (int_of_float (b *. 255.)) (int_of_float ((1. -. b) *. 255.)) 0 in
    Graphics.set_color c;
    Graphics.fill_circle x y 2
  in
  Graphics.open_graph "";
  Graphics.resize_window window window;
  List.iter (fun ((x,y),b) -> plot (x,y) b) moons;
  (* train (); *)
  Graphics.loop_at_exit [Button_down; Key_pressed] (fun _ -> raise Exit)
