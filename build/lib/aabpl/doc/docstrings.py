arguments_to_replace = list({
'<y_coord_name>':(
"""x (numpy.array):
    Array of x/longtitude coordinates"""
)
    
}.items())

def fixdocstring(func):
    # for k,v in arguments_to_replace:
    #     func.__doc__ = func.__doc__.replace(k, v)
    return func
