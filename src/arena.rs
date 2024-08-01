use std::{
    marker::PhantomData,
    num::{NonZeroU32, Wrapping},
};

pub struct Arena<T> {
    slots: Vec<Slot<T>>,
    free: Option<Handle<T>>,
}

pub struct Handle<T> {
    index: NonZeroU32,
    generation: Generation,
    _phantom: PhantomData<*const T>,
}

impl<T> Copy for Handle<T> {}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<T> Eq for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.index(), self.generation)
    }
}

type Generation = Wrapping<u32>;

impl<T> Handle<T> {
    fn index(&self) -> usize {
        (self.index.get() - 1) as usize
    }
}

enum Slot<T> {
    Free { next: Option<Handle<T>> },
    Occupied { value: T, generation: Generation },
}

impl<T> Arena<T> {
    pub const fn new() -> Arena<T> {
        Arena { slots: Vec::new(), free: None }
    }

    #[allow(dead_code)]
    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.insert_with_handle(|_| value)
    }

    pub fn insert_with_handle(&mut self, constructor: impl FnOnce(Handle<T>) -> T) -> Handle<T> {
        if let Some(mut free) = self.free {
            free.generation += 1;

            let value = constructor(free);

            let old = std::mem::replace(
                &mut self.slots[free.index()],
                Slot::Occupied { value, generation: free.generation },
            );

            let Slot::Free { next } = old else { unreachable!("occupied slot in free list") };
            self.free = next;

            return free;
        }

        let handle = Handle {
            index: NonZeroU32::new((self.slots.len() + 1) as u32).unwrap(),
            generation: Wrapping(0),
            _phantom: PhantomData,
        };

        let value = constructor(handle);
        self.slots.push(Slot::Occupied { value, generation: Wrapping(0) });

        handle
    }

    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        match std::mem::replace(&mut self.slots[handle.index()], Slot::Free { next: self.free }) {
            Slot::Free { next } => {
                self.slots[handle.index()] = Slot::Free { next };
                None
            },
            Slot::Occupied { value, generation } => {
                if handle.generation != generation {
                    self.slots[handle.index()] = Slot::Occupied { value, generation };
                    return None;
                }

                self.free = Some(handle);

                Some(value)
            },
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        match &self.slots[handle.index()] {
            Slot::Free { .. } => None,
            Slot::Occupied { value, generation } => {
                if handle.generation != *generation {
                    return None;
                }
                Some(value)
            },
        }
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        match &mut self.slots[handle.index()] {
            Slot::Free { .. } => None,
            Slot::Occupied { value, generation } => {
                if handle.generation != *generation {
                    return None;
                }
                Some(value)
            },
        }
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::ops::Index<Handle<T>> for Arena<T> {
    type Output = T;

    fn index(&self, handle: Handle<T>) -> &Self::Output {
        self.get(handle).expect("slot has been removed")
    }
}

impl<T> std::ops::IndexMut<Handle<T>> for Arena<T> {
    fn index_mut(&mut self, handle: Handle<T>) -> &mut Self::Output {
        self.get_mut(handle).expect("slot has been removed")
    }
}
