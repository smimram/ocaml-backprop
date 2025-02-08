let failwith fmt = Printf.ksprintf (fun s -> failwith s) fmt

module Pair = struct
  let map (f, g) (x1,x2) = (f x1, g x2)

  let map_left f (x1,x2) = (f x1, x2)

  let map_right g (x1,x2) = (x1, g x2)
end
