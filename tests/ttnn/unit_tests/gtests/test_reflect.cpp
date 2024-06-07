// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <reflect>

enum E { A, B };
struct foo { int a; E b; };

constexpr auto f = foo{.a = 42, .b = B};

// Compile-time checks using static_assert
static_assert(reflect::size(f) == 2, "Reflect size check failed");
static_assert(reflect::type_id(f.a) != reflect::type_id(f.b), "Reflect type_id check failed");
static_assert("foo"sv == reflect::type_name(f), "Reflect type_name (obj) check failed");
static_assert("int"sv == reflect::type_name(f.a), "Reflect type_name (a) check failed");
static_assert("E"sv == reflect::type_name(f.b), "Reflect type_name (b) check failed");
static_assert("B"sv == reflect::enum_name(f.b), "Reflect enum_name check failed");
static_assert("a"sv == reflect::member_name<0>(f), "Reflect member_name<0> check failed");
static_assert("b"sv == reflect::member_name<1>(f), "Reflect member_name<1> check failed");
static_assert(42 == reflect::get<0>(f), "Reflect get<0> check failed");
static_assert(B == reflect::get<1>(f), "Reflect get<1> check failed");
static_assert(42 == reflect::get<"a">(f), "Reflect get<\"a\"> check failed");
static_assert(B == reflect::get<"b">(f), "Reflect get<\"b\"> check failed");

constexpr auto t = reflect::to<std::tuple>(f);
static_assert(42 == std::get<0>(t), "Reflect to<std::tuple> get<0> check failed");
static_assert(B == std::get<1>(t), "Reflect to<std::tuple> get<1> check failed");

// Runtime checks using Google Test
TEST(ReflectTest, Size) {
    EXPECT_EQ(reflect::size(f), 2);
}

TEST(ReflectTest, TypeId) {
    EXPECT_NE(reflect::type_id(f.a), reflect::type_id(f.b));
}

TEST(ReflectTest, TypeName) {
    EXPECT_EQ(reflect::type_name(f), "foo"sv);
    EXPECT_EQ(reflect::type_name(f.a), "int"sv);
    EXPECT_EQ(reflect::type_name(f.b), "E"sv);
}

TEST(ReflectTest, EnumName) {
    EXPECT_EQ(reflect::enum_name(f.b), "B"sv);
}

TEST(ReflectTest, MemberName) {
    EXPECT_EQ(reflect::member_name<0>(f), "a"sv);
    EXPECT_EQ(reflect::member_name<1>(f), "b"sv);
}

TEST(ReflectTest, GetByIndex) {
    EXPECT_EQ(reflect::get<0>(f), 42);
    EXPECT_EQ(reflect::get<1>(f), B);
}

TEST(ReflectTest, GetByName) {
    EXPECT_EQ(reflect::get<"a">(f), 42);
    EXPECT_EQ(reflect::get<"b">(f), B);
}

TEST(ReflectTest, ToTuple) {
    constexpr auto t = reflect::to<std::tuple>(f);
    EXPECT_EQ(std::get<0>(t), 42);
    EXPECT_EQ(std::get<1>(t), B);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
