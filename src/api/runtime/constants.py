"""Two reserved names."""

#: The virtual module that stands for the caller, outside the mind.
#: Emit to it with `ctx.emit(..., to=WORLD)` to hand something back.
WORLD = "world"

#: Matches any source or any kind in a route.
WILDCARD = "*"
