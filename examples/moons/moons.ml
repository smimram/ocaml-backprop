open Backprop

let () = Printexc.record_backtrace true

module List = struct
  include List

  let shuffle l =
    l
    |> List.map (fun x -> Random.bits (), x)
    |> List.sort Stdlib.compare
    |> List.map snd

  let count p l =
    let n = ref 0 in
    List.iter (fun x -> if p x then incr n) l;
    !n
end

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
    Backpropagatable.Vector.neural_network ~activation:`ReLU ~weights ~bias
  in
  let layer2 =
    let weights = ref @@ Vector.Linear.uniform 16 16 in
    let bias = ref @@ Vector.zero 16 in
    Backpropagatable.Vector.neural_network ~activation:`ReLU ~weights ~bias
  in
  let layer3 =
    let weights = ref @@ Vector.Linear.uniform 16 1 in
    let bias = ref @@ Vector.zero 1 in
    Backpropagatable.Vector.neural_network ~activation:`ReLU ~weights ~bias
  in
  fun x -> Vector.pair x |> Backpropagatable.cst |> layer1 |> layer2 |> layer3

let train () =
  let steps = 100 in
  for step = 0 to steps - 1 do
    List.iter
      (fun (p,b) ->
         let out =
           p
           |> net
           |> Backpropagatable.Vector.squared_distance_to (Vector.scalar b)
           |> Backpropagatable.descent 0.1
         in
         Backpropagatable.run out
      ) (List.shuffle moons);
    let predict p =
      net p |> Backpropagatable.eval |> Vector.to_scalar |> (fun x -> if x >= 0.5 then 1. else 0.)
    in
    let acc = (float @@ List.count (fun (p,b) -> predict p = b) moons) /. float (List.length moons) *. 100. in
    Printf.printf "Step %d, accuracy: %.00f%%\n%!" step acc
  done

let () =
  let display = ref false in
  Arg.parse
    [
      "--display", Arg.Set display, "Provide graphical representation."
    ]
    (fun _ -> ())
    "moons [options]";
  Random.self_init ();
  let window = 500 in
  let plot (x,y) b =
    let x = x *. float window |> int_of_float in
    let y = y *. float window |> int_of_float in
    let c = Graphics.rgb (int_of_float ((1. -. b) *. 255.)) (int_of_float (b *. 255.)) 0 in
    Graphics.set_color c;
    Graphics.fill_circle x y 2
  in
  if !display then
    (
      Graphics.open_graph "";
      Graphics.resize_window window window;
      List.iter (fun ((x,y),b) -> plot (x,y) b) moons;
    );
  train ();
  if !display then
    Graphics.loop_at_exit [Button_down; Key_pressed] (fun _ -> raise Exit)
