# pystk-gym
A gym env for the game SuperTuxKart using pystk.

Here is what I would like to do for a VAE:

* don't calculate info (waste of computation)
* ability to save photos locally
* human controllable kart and env

I should be able to hook up a human controller to the env. check what methods i need to change

One idea i have now is to:

* limit it to 1 kart if it's a human controllable env
* init env with a plt window and on render function call, I would render the image to the plt
  window and on step i would get the actions from the plt window. What would be a better way to get
  the actions as a param instead of calculationg it myself inside the env (take a look at
  highway_env)
