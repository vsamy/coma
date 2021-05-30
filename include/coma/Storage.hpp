/*
 * Copyright 2020-2021 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

namespace coma {

/*! \brief coma underlying storage class. */
template <typename T, int MaxLength>
struct Storage {
    COMA_STATIC_ASSERT(MaxLength >= 0, "Storage length must be positive or Dynamic");
    using underlying_t = std::array<T, MaxLength>;
    underlying_t values;

    Storage() = default;
    Storage(Index) noexcept {};
    Storage(const T& defaultValue) { values.fill(defaultValue); }
    template <typename... Args, typename = std::enable_if_t<!(std::is_same_v<Storage, std::decay_t<Args>> && ...)>>
    constexpr void set(Args&&... args)
    {
        COMA_STATIC_ASSERT(sizeof...(Args) == MaxLength, "Number of parameters must be equal to MaxLength!");
        COMA_STATIC_ASSERT((std::is_constructible_v<T, Args&&> && ...), "All parameters must be able to construct a type T!");

        auto itr = values.begin();
        ((*(itr++) = T{ std::forward<Args>(args) }), ...);
    }
    constexpr void set(underlying_t other) noexcept { values = std::move(other); }

    constexpr void swap(Storage& other) noexcept { values.swap(other.values); }
    constexpr void resize(Index s [[maybe_unused]]) noexcept { COMA_ASSERT(static_cast<Index>(MaxLength) == s, "This is a fixed-size storage and thus non-resizable, s must be equal to MaxLength"); } // do nothing
    constexpr Index size() const noexcept { return static_cast<Index>(MaxLength); }
    constexpr T* data() noexcept { return values.data(); }
    constexpr const T* data() const noexcept { return values.data(); }
    constexpr T& operator[](Index pos) noexcept { return values[static_cast<size_t>(pos)]; }
    constexpr const T& operator[](Index pos) const noexcept { return values[static_cast<size_t>(pos)]; }
    constexpr T& at(Index pos) { return values.at(static_cast<size_t>(pos)); }
    constexpr const T& at(Index pos) const { return values.at(static_cast<size_t>(pos)); }

    friend bool operator==(const Storage& lhs, const Storage& rhs) noexcept { return lhs.values == rhs.values; }
    friend bool operator!=(const Storage& lhs, const Storage& rhs) noexcept { return !(lhs == rhs); }
};

template <typename T>
struct Storage<T, Dynamic> {
    using underlying_t = std::vector<T>;
    underlying_t values;

    Storage() = default;
    Storage(Index size)
        : values(static_cast<size_t>(size))
    {
        COMA_ASSERT(size >= 0, "Length must be positive");
    }
    Storage(Index size, const T& defaultValue)
        : values(static_cast<size_t>(size), defaultValue)
    {
        COMA_ASSERT(size >= 0, "Length must be positive");
    }
    template <typename... Args, typename = std::enable_if_t<!(std::is_same_v<Storage, std::decay_t<Args>> && ...)>>
    void set(Args&&... args)
    {
        COMA_STATIC_ASSERT((std::is_constructible_v<T, Args&&> && ...), "All parameters must be able to construct a type T!");

        values.resize(sizeof...(Args));
        auto itr = values.begin();
        ((*(itr++) = T{ std::forward<Args>(args) }), ...);
    }
    void set(underlying_t other) noexcept { values = std::move(other); }

    void swap(Storage& other) noexcept { values.swap(other.values); }
    void resize(Index length) { 
        COMA_ASSERT(length >= 0, "Length must be positive");
        values.resize(static_cast<size_t>(length));
    }
    Index size() const noexcept { return static_cast<Index>(values.size()); }
    T* data() noexcept { return values.data(); }
    const T* data() const noexcept { return values.data(); }
    T& operator[](Index pos) noexcept { return values[static_cast<size_t>(pos)]; }
    const T& operator[](Index pos) const noexcept { return values[static_cast<size_t>(pos)]; }
    T& at(Index pos) { return values.at(static_cast<size_t>(pos)); }
    const T& at(Index pos) const { return values.at(static_cast<size_t>(pos)); }

    friend bool operator==(const Storage& lhs, const Storage& rhs) noexcept { return lhs.values == rhs.values; }
    friend bool operator!=(const Storage& lhs, const Storage& rhs) noexcept { return !(lhs == rhs); }
};

} // namespace coma
