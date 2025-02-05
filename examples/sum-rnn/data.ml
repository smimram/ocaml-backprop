let generate_sums dataset_size sequence_length max_num =
  
  let x = Array.init (dataset_size*sequence_length)
    (fun _ -> Random.int max_num) in
  let y = Array.init dataset_size 
    (fun n -> 
      let start = n*sequence_length in
      let s = ref 0 in
      for i = 0 to sequence_length-1 do
        s := !s + x.(start+i)
      done;
      !s
      ) in
  x, y

let pad n s =
  s ^ String.make (n - String.length s) ' '

let int_lenght n =
  String.length @@ string_of_int n

let generate_string_data dataset_size sequence_length max_num =
  let x, y = generate_sums dataset_size sequence_length max_num in
  let max_size_x = sequence_length - 1 + 
    (int_lenght max_num)*sequence_length in
  let x = Array.init dataset_size
    (fun n -> 
      let start = n*sequence_length in
      let s = ref @@ string_of_int x.(start) in
      for i = 1 to sequence_length-1 do
        s := !s ^ "+" ^ string_of_int x.(start+i)
      done;
      pad max_size_x !s
      ) in
  let max_size_y = int_lenght max_num*sequence_length in
  let y = Array.map 
      (fun x ->
        pad max_size_y @@ string_of_int x
        ) 
      y in
  x,y
  
